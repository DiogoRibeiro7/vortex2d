from .vortex2d import (
    VortexSystem2D,
    PassiveTracers2D,
    VortexState,
    AnimationConfig,
    plot_snapshot,
    run_animation,
)
from .plotly_viz import (
    plot_snapshot_interactive,
    run_animation_interactive,
    PlotlySnapshotConfig,
)

# Backwards-compatible re-export
from .vortex2d import (
    VortexSystem2D,
    PassiveTracers2D,
    VortexState,
    AnimationConfig,
    plot_snapshot,
    run_animation,
)
from .api import (
    save_npz, load_npz,
    SimulationConfig, NumericsConfig,
)
from .seeders import (
    seed_gaussian_cloud, seed_shear_layer, seed_vortex_ring,
)

# Backwards-compatible re-export
from .vortex2d import (
    VortexSystem2D,
    PassiveTracers2D,
    VortexState,
    PeriodicFFTConfig,
    TreecodeConfig,
    PSEConfig,
    AnimationConfig,
    plot_snapshot,
    run_animation,
)

__all__ = [
    "plot_snapshot_interactive", "run_animation_interactive", "PlotlySnapshotConfig",
    "save_npz", "load_npz", "SimulationConfig", "NumericsConfig",
    "seed_gaussian_cloud", "seed_shear_layer", "seed_vortex_ring",
    "VortexSystem2D",
    "PassiveTracers2D",
    "VortexState",
    "PeriodicFFTConfig",
    "TreecodeConfig",
    "PSEConfig",
    "AnimationConfig",
    "plot_snapshot",
    "run_animation",
]
