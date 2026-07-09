// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Tests for MERGE INTO planning: verify the `TableProvider::merge_into` hook
//! receives the planned USING source, the ON condition, and the WHEN clauses.

use std::sync::{Arc, Mutex};

use arrow::array::{Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::datasource::{MemTable, TableProvider, TableType};
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::Expr;
use datafusion_catalog::Session;
use datafusion_common::{DFSchemaRef, TableReference};
use datafusion_expr::dml::{MergeIntoAction, MergeIntoClause, MergeIntoClauseKind};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::empty::EmptyExec;

/// Everything the merge_into() hook received, captured for assertions.
#[derive(Clone)]
struct CapturedMerge {
    source_schema: SchemaRef,
    source_df_schema: DFSchemaRef,
    target_ref: TableReference,
    on: Expr,
    clauses: Vec<MergeIntoClause>,
}

/// A TableProvider that captures the arguments passed to merge_into().
struct CaptureMergeProvider {
    schema: SchemaRef,
    received: Arc<Mutex<Option<CapturedMerge>>>,
}

impl CaptureMergeProvider {
    fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            received: Arc::new(Mutex::new(None)),
        }
    }

    fn captured(&self) -> Option<CapturedMerge> {
        self.received.lock().unwrap().clone()
    }
}

impl std::fmt::Debug for CaptureMergeProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CaptureMergeProvider")
            .field("schema", &self.schema)
            .finish()
    }
}

#[async_trait]
impl TableProvider for CaptureMergeProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(EmptyExec::new(Arc::clone(&self.schema))))
    }

    async fn merge_into(
        &self,
        _state: &dyn Session,
        source: Arc<dyn ExecutionPlan>,
        source_schema: DFSchemaRef,
        target_ref: TableReference,
        on: Expr,
        clauses: Vec<MergeIntoClause>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        *self.received.lock().unwrap() = Some(CapturedMerge {
            source_schema: source.schema(),
            source_df_schema: source_schema,
            target_ref,
            on,
            clauses,
        });
        Ok(Arc::new(EmptyExec::new(Arc::new(Schema::new(vec![
            Field::new("count", DataType::UInt64, false),
        ])))))
    }
}

fn target_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]))
}

/// An SCD2-shaped target: business columns + validity interval + change metadata.
fn scd2_target_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
        Field::new("_valid_from", DataType::Int64, false),
        Field::new("_valid_to", DataType::Int64, true),
        Field::new("_is_current", DataType::Boolean, false),
        Field::new("_cdc_offset", DataType::Int64, false),
    ]))
}

fn source_batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Utf8, true),
            Field::new("_valid_from", DataType::Int64, false),
            Field::new("_cdc_offset", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(Int64Array::from(vec![10, 20, 10])),
            Arc::new(Int64Array::from(vec![100, 101, 102])),
        ],
    )
    .unwrap()
}

fn setup(target: SchemaRef) -> (SessionContext, Arc<CaptureMergeProvider>) {
    let ctx = SessionContext::new();
    let provider = Arc::new(CaptureMergeProvider::new(target));
    ctx.register_table("t", Arc::clone(&provider) as Arc<dyn TableProvider>)
        .unwrap();
    let batch = source_batch();
    let mem = MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap();
    ctx.register_table("batch", Arc::new(mem)).unwrap();
    (ctx, provider)
}

#[tokio::test]
async fn merge_into_hook_receives_on_and_clauses() -> Result<()> {
    let (ctx, provider) = setup(target_schema());

    ctx.sql(
        "MERGE INTO t USING batch s ON t.id = s.id \
         WHEN MATCHED THEN UPDATE SET val = s.val \
         WHEN NOT MATCHED THEN INSERT (id, val) VALUES (s.id, s.val)",
    )
    .await?
    .collect()
    .await?;

    let captured = provider.captured().expect("merge_into was not invoked");
    assert_eq!(captured.on.to_string(), "t.id = s.id");
    assert_eq!(captured.clauses.len(), 2);
    assert_eq!(captured.clauses[0].kind, MergeIntoClauseKind::Matched);
    assert!(matches!(
        &captured.clauses[0].action,
        MergeIntoAction::Update(assignments) if assignments.len() == 1
            && assignments[0].0 == "val"
    ));
    assert_eq!(captured.clauses[1].kind, MergeIntoClauseKind::NotMatched);
    assert!(matches!(
        &captured.clauses[1].action,
        MergeIntoAction::Insert { columns, values }
            if columns == &["id".to_string(), "val".to_string()] && values.len() == 2
    ));
    // The planned source is the bare table: its schema is the batch schema.
    assert_eq!(captured.source_schema.fields().len(), 4);
    // The logical source schema carries the USING alias as qualifier, and the
    // target reference is the canonical registered name.
    assert_eq!(
        captured.source_df_schema.qualified_field(0).0,
        Some(&TableReference::bare("s"))
    );
    assert_eq!(captured.target_ref, TableReference::bare("t"));
    Ok(())
}

/// The SCD2 upsert shape: a window-function subquery as the USING source, a
/// multi-condition ON, a column-subset UPDATE, and a clause predicate.
#[tokio::test]
async fn merge_into_subquery_source_with_window_functions() -> Result<()> {
    let (ctx, provider) = setup(scd2_target_schema());

    ctx.sql(
        "MERGE INTO t USING ( \
             SELECT id, val, _valid_from, _cdc_offset, \
                    ROW_NUMBER() OVER (PARTITION BY id ORDER BY _cdc_offset DESC) AS rn, \
                    LEAD(_valid_from) OVER (PARTITION BY id ORDER BY _valid_from) AS next_vf \
             FROM batch) s \
         ON t.id = s.id AND t._valid_from = s._valid_from AND t._cdc_offset = s._cdc_offset \
         WHEN MATCHED AND s.rn = 1 THEN UPDATE SET _valid_to = s.next_vf, _is_current = false \
         WHEN NOT MATCHED THEN INSERT (id, val, _valid_from, _valid_to, _is_current, _cdc_offset) \
             VALUES (s.id, s.val, s._valid_from, s.next_vf, true, s._cdc_offset)",
    )
    .await?
    .collect()
    .await?;

    let captured = provider.captured().expect("merge_into was not invoked");

    // The USING subquery — window functions included — arrives fully planned:
    // the source schema exposes the window-derived columns.
    let names: Vec<&str> = captured
        .source_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(names, ["id", "val", "_valid_from", "_cdc_offset", "rn", "next_vf"]);

    // Multi-condition ON survives with target/source qualifiers intact.
    assert_eq!(
        captured.on.to_string(),
        "t.id = s.id AND t._valid_from = s._valid_from AND t._cdc_offset = s._cdc_offset"
    );

    // Clause predicate + column-subset UPDATE. The analyzer coerces the
    // predicate literal to ROW_NUMBER's UInt64 (name-preserved via alias).
    assert_eq!(captured.clauses.len(), 2);
    assert_eq!(
        captured.clauses[0]
            .predicate
            .as_ref()
            .map(|p| p.clone().unalias_nested().data.to_string()),
        Some("s.rn = UInt64(1)".to_string())
    );
    assert!(matches!(
        &captured.clauses[0].action,
        MergeIntoAction::Update(assignments) if assignments.len() == 2
            && assignments[0].0 == "_valid_to"
            && assignments[1].0 == "_is_current"
    ));
    assert!(matches!(
        &captured.clauses[1].action,
        MergeIntoAction::Insert { columns, .. } if columns.len() == 6
    ));
    Ok(())
}

#[tokio::test]
async fn merge_into_returns_count_schema() -> Result<()> {
    let (ctx, provider) = setup(target_schema());

    let df = ctx
        .sql(
            "MERGE INTO t USING batch s ON t.id = s.id \
             WHEN MATCHED THEN UPDATE SET val = s.val",
        )
        .await?;

    // The DML output is the standard single count column.
    assert_eq!(df.schema().field(0).name(), "count");
    df.collect().await?;
    assert!(provider.captured().is_some());
    Ok(())
}

/// Root-cause probe: an ON clause referencing ONLY source columns passes the
/// analyzer and reaches the hook — proving the failure above is the Dml
/// node's target-column exprs being resolved against its source-only input.
#[tokio::test]
async fn merge_into_source_only_on_probe() -> Result<()> {
    let (ctx, provider) = setup(target_schema());

    ctx.sql(
        "MERGE INTO t USING batch s ON s.id = s.id \
         WHEN MATCHED THEN UPDATE SET val = s.val",
    )
    .await?
    .collect()
    .await?;

    assert!(provider.captured().is_some());
    Ok(())
}

/// Target-alias form: alias-qualified column references are rewritten to the
/// canonical table reference at planning time, so the recorded plan is
/// self-describing (the alias exists only in the SQL text).
#[tokio::test]
async fn merge_into_target_alias_normalized() -> Result<()> {
    let (ctx, provider) = setup(target_schema());

    ctx.sql(
        "MERGE INTO t AS tgt USING batch s ON tgt.id = s.id \
         WHEN MATCHED THEN UPDATE SET val = s.val \
         WHEN NOT MATCHED THEN INSERT (id, val) VALUES (s.id, s.val)",
    )
    .await?
    .collect()
    .await?;

    let captured = provider.captured().expect("merge_into was not invoked");
    assert_eq!(captured.on.to_string(), "t.id = s.id");
    Ok(())
}
