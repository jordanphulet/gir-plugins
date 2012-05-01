tcr_data = UnserializeData( 'tcr_data.out' );
tcr_im = IFFT2D( tcr_data );
tcr_im = sqrt( sum( abs( tcr_im ).^2, 3 ) );
tcr_mov = squeeze( tcr_im(:,:,1,1,1,2,1,:) );
