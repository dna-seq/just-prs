import reflex as rx
from reflex_mui_datagrid.models import ColumnDef
from reflex.utils.format import format_state
class S(rx.State):
    cols: list[dict] = []
    def set_cols(self):
        self.cols = [ColumnDef(field="test", render_cell=rx.Var("() => 'N/A'")).dict()]

s = S()
s.set_cols()
print(format_state({s.get_name(): s.dict()}))
