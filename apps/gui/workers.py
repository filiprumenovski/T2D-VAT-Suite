import sys
import traceback
from PySide6.QtCore import QObject, Signal, Slot, QRunnable

class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)
    result = Signal(object)
    progress = Signal(str)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs, progress_callback=self.signals.progress)
            self.signals.result.emit(result)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit(str(value))
        finally:
            self.signals.finished.emit()
