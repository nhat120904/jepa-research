"""Optional Weights & Biases logging.

Mode comes from WANDB_MODE, which the Slurm scripts set to online when a key and the network are available and to
offline otherwise (sync later with `wandb sync <run>/wandb/offline-run-*` from the login node). If wandb is missing,
disabled, or cannot start, every call is a no-op and the run's JSON reports remain the record.
"""

import os


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "/"))
        elif isinstance(v, (int, float, bool, str)) or v is None:
            out[key] = v
    return out


class Logger:
    def __init__(self, run_dir, job_type, config=None, name=None):
        self.run = None
        if os.environ.get("WANDB_MODE") == "disabled":
            return
        try:
            import wandb
        except ImportError:
            return
        kwargs = dict(project=os.environ.get("WANDB_PROJECT", "cta-pusht"), entity=os.environ.get("WANDB_ENTITY"),
                      group=os.environ.get("CTA_TAG"), job_type=job_type, name=name, dir=str(run_dir),
                      config=config or {})
        try:
            self.run = wandb.init(**kwargs)
        except Exception as err:  # network or auth failure: keep the record locally
            print(f"wandb online init failed ({err}); logging offline", flush=True)
            try:
                self.run = wandb.init(mode="offline", **kwargs)
            except Exception as err2:
                print(f"wandb disabled: {err2}", flush=True)
        self.wandb = wandb

    def log(self, data, step=None):
        if self.run is not None:
            self.run.log(_flat(data), step=step)

    def summary(self, data):
        if self.run is not None:
            self.run.summary.update(_flat(data))

    def table(self, key, columns, rows):
        if self.run is not None:
            self.run.log({key: self.wandb.Table(columns=list(columns), data=[list(r) for r in rows])})

    def finish(self):
        if self.run is not None:
            self.run.finish()
