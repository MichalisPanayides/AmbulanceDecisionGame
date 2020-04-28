import ambulance_game.simulation
import ambulance_game.markov

from .timing_experiments import (
    time_for_different_number_of_trials,
    old_import_trials_duration,
    import_trials_duration,
    get_duration_distribution_plot,
    get_duration_all_lines_plot,
    get_duration_mean_plot,
    make_plot_of_confidence_intervals_over_iterations,
)

from .model_plots import (
    make_plot_for_different_thresholds,
    make_plot_for_proportion_within_target,
    make_plot_two_hospitals_arrival_split,
    make_plot_of_confidence_intervals_over_warm_up_time,
    make_plot_of_confidence_intervals_over_runtime,
)


from .comparisons import get_heatmaps
