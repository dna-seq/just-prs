import reflex as rx
from reflex_mui_datagrid.models import ColumnDef
class S(rx.State):
    cols: list[dict] = []
    def set_cols(self):
        self.cols = [ColumnDef(field="test", render_cell=rx.Var("() => 'N/A'")).dict()]

print("compiled")
