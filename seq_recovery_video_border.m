%% Sequential Recovery for Solving Fourier Phase Retrieval
% Suppose we know a thin border (one pixel) around an image. We
% sequentially estimate other columns, rows, or columns-rows
% We estimate two columns and then two rows at a time
% Suppose the image with border is m*n

%%
clear variables
clc
close all

%% Load images
data = load('person01_boxing_d2_uncomp_frames.mat');
% data = load('person01_handwaving_d2_uncomp_frames.mat');
% data = load('person10_handclapping_d2_uncomp_frames.mat');

video = data.video;

num_frames = 3;

% fig = figure(300); fig.Position = [100 100 1000 500];
% [ha, pos] = tight_subplot(2,num_frames, [0.08 0.005], [0.02 0.01], [0.005 0.005]);

% fig = figure(300); fig.Position = [100 100 1000 250];
% [ha, pos] = tight_subplot(1,num_frames, [0.08 0.005], [0.02 0.13], [0.005 0.005]);

fig = figure(300); fig.Position = [100 100 1000 320];
[ha, pos] = tight_subplot(1,num_frames, [0.08 0.005], [0.17 0.01], [0.005 0.005]);


for im = 1: num_frames
    image_org = video(3*im+1+1,:,:,:);
%     image_org = video(2*im+2,:,:,:);
%     image_org = video(2*im+1+113,:,:,:);

    image_org = im2double(image_org);
    image_org = rgb2gray(squeeze(image_org));
    I_mod = image_org;
        
    m = size(image_org,1);
    n = size(image_org,2);
    
    % Design the border
    known_border_width_columns = 15;
    known_border_width_rows = 15;
    
    c_1 = I_mod(:,1);
    c_n = I_mod(:,end);
    
    
    r_1 = I_mod(1,:);
    r_m = I_mod(end,:);
    
    R_corr = xcorr2(I_mod,I_mod);
    
    toeplitz_left = @(z) fliplr(toeplitz([z; zeros(m-1,1)], [z(1),zeros(1,m-1)]));
    toeplitz_right = @(z) toeplitz([flipud(z); zeros(m-1,1)], [z(end),zeros(1,m-1)]);
    
    toeplitz_up = @(z) fliplr(toeplitz([z; zeros(n-1,1)], [z(1),zeros(1,n-1)]));
    toeplitz_down = @(z) toeplitz([flipud(z); zeros(n-1,1)], [z(end),zeros(1,n-1)]);
    
    c_1_T = toeplitz_left(c_1);
    c_n_T = toeplitz_right(c_n);
    
    r_1_T = toeplitz_up(r_1');
    r_m_T = toeplitz_down(r_m');
    
    %% Sequential Recover
    knownRows = [1:known_border_width_rows m-known_border_width_rows+1:m];
    unknownRows = setdiff([1:m],knownRows);
    H_c = [c_1_T(:,unknownRows) c_n_T(:,unknownRows)];
    
    knownColumns = [1 :known_border_width_columns n-known_border_width_columns+1:n];
    unknownColumns = setdiff([1:n],knownColumns);
    H_r = [r_1_T(:,unknownColumns) r_m_T(:,unknownColumns)];
    
    % Recovery
    I_est = 0*I_mod;
    I_est(:,knownColumns) = I_mod(:,knownColumns);         % set known columns: first and last ones
    I_est(knownRows,:) = I_mod(knownRows,:);               % set known rows at each column: first and last pixels
    
    H_c_inv = pinv(H_c);
    cond(H_c)
    
    for k = known_border_width_columns+1:ceil(size(I_est,2)/2)
        
        
        % estimate two columns
        
        R_n = R_corr(:,k);
        
        R_res = 0;
        
        for l = 1:k-2
            % from left to right
            R_res = R_res + conv(flipud(I_est(:,end-l)),I_est(:,k-l));
            
        end
        
        if k > 2
            R_est = 0;
            for l = 0:k-2
                R_est = R_est + conv(flipud(I_est(:,end-l)),I_est(:,k-l-1));
            end
            R_error = (R_corr(:,k-1)-R_est);
        end
        
        R_h = R_n-R_res; % remove the terms from correlation of known columns
        rowTerms = c_1_T(:,knownRows)*I_est(knownRows,end-k+1) + c_n_T(:,knownRows)*I_est(knownRows,k);  % remove the terms from the known rows
        R_h = R_h - rowTerms;
        
        I_tmp = reshape(H_c_inv*R_h,[],2);
        
        I_tmp ( I_tmp > 1) = 1;
        I_tmp ( I_tmp < 0) = 0;
        
        I_est(unknownRows,end-k+1) = I_tmp(:,1);
        I_est(unknownRows,k) = I_tmp(:,2);
        
        I_est_psnr = psnr(I_est(unknownRows,unknownColumns),I_mod(unknownRows,unknownColumns));
        I_est_ssim = ssim(I_est(unknownRows,unknownColumns),I_mod(unknownRows,unknownColumns));
        
%         axes(ha(im))
%         imagesc(I_mod)
%         if (im == 1)
%             ylabel('Original','FontSize',34)
%         end
%         
%         set(gca,'xtick',[])
%         set(gca,'ytick',[])
%         
        axes(ha(im))
        imagesc(I_est);
%         if (im == 1)
%             ylabel('Recovered','FontSize',34)
%         end
%         
            xlabel(sprintf('(%0.2f, %0.2f)',I_est_psnr,I_est_ssim),'FontSize',28)
%        
        
        colormap gray
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        
        
        drawnow
        
    end
    
    
end





