import reflex as rx
from reflex_mui_datagrid.models import ColumnDef
col = ColumnDef(field="test", render_cell=rx.Var("() => 'N/A'"))
print(col.dict())
