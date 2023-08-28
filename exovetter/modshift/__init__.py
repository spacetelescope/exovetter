#Late night work around
# I want to do from .modshift import *
#but pyflakes won't let me 


from .modshift import (
    compute_modshift_metrics, 
    fold_and_bin_data,
    compute_false_alarm_threshold, 
    compute_event_significances,
    find_indices_of_key_locations, 
    mark_phase_range,
    estimate_scatter,
    compute_convolution_for_binned_data,
    compute_phase,
)


compute_modshift_metrics 
fold_and_bin_data
compute_false_alarm_threshold 
compute_event_significances
find_indices_of_key_locations
mark_phase_range
estimate_scatter
compute_convolution_for_binned_data
compute_phase


from .plotmodshift import (
    plot_modshift, 
    mark_events, 
    mark_false_alarm_threshold,
)

plot_modshift, 
mark_events, 
mark_false_alarm_threshold,
