"""
TrainerBase / SimpleTrainer
=============================
This is the generic training loop, deliberately kept independent of
*what* is being trained (MLP vs CNN, MNIST vs a future dataset). All of
that is injected: the model, the data loader, and the optimizer are
handed to SimpleTrainer's constructor; SimpleTrainer just knows the
mechanics of "pull a batch, forward, backward, step".

ITERATION-BASED, NOT EPOCH-BASED (a deliberate departure from your
original v1 script, matching Detectron2's convention): Detectron2 trains
for `max_iter` steps, not `max_epochs`. We follow that here -- the
DataLoader is treated as an infinite stream (re-created when exhausted),
and epoch boundaries are just `len(train_loader)` iterations apart. This
lets hooks (checkpointing, eval, logging) all be expressed uniformly as
"every N iterations", which is exactly how Detectron2's hooks work.
tools/train_net.py converts your familiar `max_epochs` into `max_iter`
once, at the top, so config files can still be written in epochs.

THE HOOK SYSTEM: hooks are the extension point. Instead of hardcoding
"save a checkpoint every epoch" and "log every N steps" inside the loop
itself, the loop only calls four generic extension points --
before_train / before_step / after_step / after_train -- and any number
of Hook objects can attach behavior to them. Adding new behavior later
(e.g. a learning-rate warmup hook) never requires touching this file.
"""
import logging

logger = logging.getLogger(__name__)


class HookBase:
    """
    Base class for a training hook. Override any subset of these four
    methods. `self.trainer` is set automatically when the hook is
    registered with a trainer (see TrainerBase.register_hooks).
    """

    trainer = None

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass


class TrainerBase:
    """
    Generic iteration loop + hook dispatch. Subclasses only need to
    implement `run_step()` (one iteration of forward/backward/optimize).
    """

    def __init__(self):
        self._hooks = []
        self.iter = 0
        self.start_iter = 0
        self.max_iter = 0

    def register_hooks(self, hooks):
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            h.trainer = self
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        logger.info(f"Starting training from iteration {start_iter} to {max_iter}")
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        self._call_hooks("before_train")
        try:
            for self.iter in range(start_iter, max_iter):
                self._call_hooks("before_step")
                self.run_step()
                self._call_hooks("after_step")
        finally:
            self._call_hooks("after_train")

    def _call_hooks(self, event: str):
        for h in self._hooks:
            getattr(h, event)()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    The standard supervised-classification training step:
    pull a (images, targets) batch, forward through the model (which
    returns a loss dict in training mode -- see modeling/meta_arch/),
    sum the losses, backward, optimizer step.
    """

    def __init__(self, model, data_loader, optimizer):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.latest_losses = {}

    def _next_batch(self):
        try:
            return next(self._data_loader_iter)
        except StopIteration:
            # Epoch boundary: the loader is exhausted, start it again.
            self._data_loader_iter = iter(self.data_loader)
            return next(self._data_loader_iter)

    def run_step(self):
        self.model.train()
        images, targets = self._next_batch()

        loss_dict = self.model(images, targets)
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        self.latest_losses = {k: v.detach().item() for k, v in loss_dict.items()}
        self.latest_losses["total_loss"] = losses.detach().item()
