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
images = {'baboon.png','cameraman.tif','coins.png'};
measurement_noise_snr = 0:5:80;
known_border_width = 8;
rng(0)
for im = 1: length(images)
    
    fig = figure; fig.Position = [100 100 1150 310];
    [ha, pos] = tight_subplot(1,4, [0.005 0.005], [0.175 0.005], [0.005 0.005]);
    
    if (im == 1)
        fig = figure; fig.Position = [100 100 1150 350];
        [ha, pos] = tight_subplot(1,4, [0.005 0.005], [0.15 0.13], [0.005 0.005]);
    end
    
    image_name = images{im};
    image_org = im2double(imread(image_name));
    image_org = imresize(image_org,[64 64]);
    
    axes(ha(1))
    imagesc(image_org);
    g = 2;
    if (im == 1)
        title('Original','FontSize',28)
        xlabel('(PSNR, SSIM)','FontSize',30)
    end
    colormap gray
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    
    for jj = 1: length(measurement_noise_snr)
        
        if (im == 1)
            I_est_all_channel = zeros(size(image_org,1)+2*known_border_width,size(image_org,2) + 2* known_border_width,3);
            I_original = zeros(size(image_org,1)+2*known_border_width,size(image_org,2) + 2* known_border_width,3);
            for c = 1: 3
                
                image_org_channel = image_org(:,:,c);
                
                m_org = size(image_org_channel,1);
                n_org = size(image_org_channel,2);
                
                % Design the border
                I_mod = padarray(image_org_channel,[known_border_width known_border_width],0,'both');
                
                I_mod(1,1) = 64;
                I_mod(1,end) = 64;
                [m,n] = size(I_mod);
                
                known_unknown_ratio = (m*n)/(m_org*n_org)-1
                
                c_1 = I_mod(:,1);
                c_n = I_mod(:,end);
                
                r_1 = I_mod(1,:);
                r_m = I_mod(end,:);
                
                R_corr = awgn(xcorr2(I_mod,I_mod),measurement_noise_snr(jj),'measured');
                R_noise = R_corr - xcorr2(I_mod,I_mod);
                R_org = xcorr2(I_mod,I_mod);
                estimated_snr(jj) = 20*log10(norm(R_org(:))/norm(R_noise(:)));
                
                toeplitz_left = @(z) fliplr(toeplitz([z; zeros(m-1,1)], [z(1),zeros(1,m-1)]));
                toeplitz_right = @(z) toeplitz([flipud(z); zeros(m-1,1)], [z(end),zeros(1,m-1)]);
                
                toeplitz_up = @(z) fliplr(toeplitz([z; zeros(n-1,1)], [z(1),zeros(1,n-1)]));
                toeplitz_down = @(z) toeplitz([flipud(z); zeros(n-1,1)], [z(end),zeros(1,n-1)]);
                
                c_1_T = toeplitz_left(c_1);
                c_n_T = toeplitz_right(c_n);
                
                r_1_T = toeplitz_up(r_1');
                r_m_T = toeplitz_down(r_m');
                
                %% Sequential Recover
                knownRows = [1:known_border_width m-known_border_width+1:m];
                unknownRows = setdiff([1:m],knownRows);
                H_c = [c_1_T(:,unknownRows) c_n_T(:,unknownRows)];
                
                knownColumns = [1 :known_border_width n-known_border_width+1:n];
                unknownColumns = setdiff([1:n],knownColumns);
                H_r = [r_1_T(:,unknownColumns) r_m_T(:,unknownColumns)];
                
                % Recovery
                I_est = 0*I_mod;
                I_est(:,knownColumns) = I_mod(:,knownColumns);        % set known columns: first and last ones
                I_est(knownRows,:) = I_mod(knownRows,:);               % set known rows at each column: first and last pixels
                
                H_c_inv = pinv(H_c);
                
                for k = 2:ceil(size(I_est,2)/2)
                    
                    if (k >= known_border_width+1)
                        
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
                        I_tmp ( I_tmp <- 1) = -1;
                        
                        I_est(unknownRows,end-k+1) = I_tmp(:,1);
                        I_est(unknownRows,k) = I_tmp(:,2);
                        
                    end
                end
                
                I_est_all_channel(:,:,c) = I_est;
                I_original(:,:,c) = I_mod;
                
            end
            
            I_est_psnr = psnr(I_est_all_channel(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width,:),I_original(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width,:));
            I_est_ssim = ssim(I_est_all_channel(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width,:),I_original(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width,:));
            
            
            if (measurement_noise_snr(jj) == 30 || measurement_noise_snr(jj) == 40 || measurement_noise_snr(jj) == 50)
                axes(ha(6-g))
                imagesc(I_est_all_channel(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width,:));
                title(sprintf('SNR = %d dB',measurement_noise_snr(jj)),'FontSize',30)
                xlabel(sprintf('(%0.2f, %0.2f)',I_est_psnr,I_est_ssim),'FontSize',28)
                
                colormap gray
                set(gca,'xtick',[])
                set(gca,'ytick',[])
                g = g + 1;
                saveas(gcf,sprintf('Recovered_Known_Border_Noisy (im = %d).png',im))
            end
            
            psnr_all(jj,im) = I_est_psnr;
            ssim_all(jj,im) = I_est_ssim;
            
            
        else
            m_org = size(image_org,1);
            n_org = size(image_org,2);
            I_mod = padarray(image_org,[known_border_width known_border_width],0,'both');
            
            I_mod(1,1) = 64;
            I_mod(1,end) = 64;
            [m,n] = size(I_mod);
            
            known_unknown_ratio = (m*n)/(m_org*n_org)-1
            
            c_1 = I_mod(:,1);
            c_n = I_mod(:,end);
            
            r_1 = I_mod(1,:);
            r_m = I_mod(end,:);
            
            R_corr = awgn(xcorr2(I_mod,I_mod),measurement_noise_snr(jj),'measured');
            R_noise = R_corr - xcorr2(I_mod,I_mod);
            R_org = xcorr2(I_mod,I_mod);
            estimated_snr(jj) = 20*log10(norm(R_org(:))/norm(R_noise(:)));
            
            toeplitz_left = @(z) fliplr(toeplitz([z; zeros(m-1,1)], [z(1),zeros(1,m-1)]));
            toeplitz_right = @(z) toeplitz([flipud(z); zeros(m-1,1)], [z(end),zeros(1,m-1)]);
            
            toeplitz_up = @(z) fliplr(toeplitz([z; zeros(n-1,1)], [z(1),zeros(1,n-1)]));
            toeplitz_down = @(z) toeplitz([flipud(z); zeros(n-1,1)], [z(end),zeros(1,n-1)]);
            
            c_1_T = toeplitz_left(c_1);
            c_n_T = toeplitz_right(c_n);
            
            r_1_T = toeplitz_up(r_1');
            r_m_T = toeplitz_down(r_m');
            
            %% Sequential Recover
            knownRows = [1:known_border_width m-known_border_width+1:m];
            unknownRows = setdiff([1:m],knownRows);
            H_c = [c_1_T(:,unknownRows) c_n_T(:,unknownRows)];
            
            knownColumns = [1 :known_border_width n-known_border_width+1:n];
            unknownColumns = setdiff([1:n],knownColumns);
            H_r = [r_1_T(:,unknownColumns) r_m_T(:,unknownColumns)];
            
            % Recovery
            I_est = 0*I_mod;
            I_est(:,knownColumns) = I_mod(:,knownColumns);        % set known columns: first and last ones
            I_est(knownRows,:) = I_mod(knownRows,:);               % set known rows at each column: first and last pixels
            
            H_c_inv = pinv(H_c);
            
            for k = 2:ceil(size(I_est,2)/2)
                
                if (k >= known_border_width+1)
                    
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
                    I_tmp ( I_tmp <- 1) = -1;
                    
                    I_est(unknownRows,end-k+1) = I_tmp(:,1);
                    I_est(unknownRows,k) = I_tmp(:,2);
                    
                end
                
                I_est_psnr = psnr(I_est(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width),I_mod(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width));
                I_est_ssim = ssim(I_est(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width),I_mod(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width));
                
            end
            
            if (measurement_noise_snr(jj) == 30 || measurement_noise_snr(jj) == 40 || measurement_noise_snr(jj) == 50)
                axes(ha(6-g))
                imagesc(I_est(known_border_width+1:end-known_border_width,known_border_width+1:end-known_border_width));
                xlabel(sprintf('(%0.2f, %0.2f)',I_est_psnr,I_est_ssim),'FontSize',28)
                
                colormap gray
                set(gca,'xtick',[])
                set(gca,'ytick',[])
                g = g + 1;
                saveas(gcf,sprintf('Recovered_Known_Border_Noisy (im = %d).png',im))
            end
        end
        psnr_all(jj,im) = I_est_psnr;
        ssim_all(jj,im) = I_est_ssim;
        
    end
end
%% PSNR and SSIM

figure
plot(measurement_noise_snr,psnr_all(:,1),'-*','Linewidth',2.5)
hold all
plot(measurement_noise_snr,psnr_all(:,2),'-o','Linewidth',2.5)
hold all
plot(measurement_noise_snr,psnr_all(:,3),'-^','Linewidth',2.5)
grid on
grid minor
xlabel('SNR (dB)','FontSize',18)
ylabel('PSNR (dB)','FontSize',18)

legend('Baboon','Cameraman','Coins','FontSize',20)
ax = gca ;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;


figure
plot(measurement_noise_snr,ssim_all(:,1),'-*','Linewidth',2.5)
hold all
plot(measurement_noise_snr,ssim_all(:,2),'-o','Linewidth',2.5)
hold all
plot(measurement_noise_snr,ssim_all(:,3),'-^','Linewidth',2.5)
grid on
grid minor
xlabel('SNR (dB)','FontSize',18)
ylabel('SSIM','FontSize',18)

legend('Baboon','Cameraman','Coins','FontSize',20)

ax = gca ;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;




