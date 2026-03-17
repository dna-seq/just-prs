import reflex as rx
from reflex_mui_datagrid.models import ColumnDef

def get_cols():
    return [
        ColumnDef(
            field="pct_AFR",
            header_name="AFR",
            type="number",
            hide=rx.Var("!state.compute_all_populations"),
        )
    ]
print(get_cols()[0].dict())
