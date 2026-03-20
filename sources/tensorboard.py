from torch.utils.tensorboard import SummaryWriter


# Own Tensorboard class giving ability to use single writer across multiple training steps.
# Allows us also to easily log additional data.
class TensorBoard:

    def __init__(self, log_dir):
        self.step = 1
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)

    # Called at end of each training step (mirrors old Keras Callback interface)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(self.step, **(logs or {}))

    # Custom method for saving own metrics (can be called externally)
    def update_stats(self, step, **stats):
        self._write_logs(stats, step)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            self.writer.add_scalar(name, float(value), index)
        self.writer.flush()
