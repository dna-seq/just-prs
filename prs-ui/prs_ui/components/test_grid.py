import reflex as rx
from reflex_mui_datagrid import data_grid, ColumnDef
class TestState(rx.State):
    rows: list[dict] = [{"id": 1, "val": 50}, {"id": 2, "val": None}]
    cols: list[dict] = [ColumnDef(field="val", header_name="Val", render_cell=rx.Var("((params) => { return <span style={{color: 'red'}}>{params.value || 'N/A'}</span>; })")).dict()]
def test_page():
    return data_grid(rows=TestState.rows, columns=TestState.cols, row_id_field="id")
