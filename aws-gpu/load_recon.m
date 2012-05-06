tcr_data = UnserializeData( '/DATA/aws-data/tcr_out.ser' );
tcr_im = sqrt( sum( abs( tcr_data ).^2, 3 ) );
tcr_im = flipdim( tcr_im, 1 );
tcr_mov = squeeze( tcr_im(:,:,1,1,1,2,1,:) );
