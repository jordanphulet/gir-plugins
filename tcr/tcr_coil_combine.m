function [result] = tcr_combine_channels( mri_data, params, meas_data )
	result = single( abs( sum( mri_data, 3 ) ) );
