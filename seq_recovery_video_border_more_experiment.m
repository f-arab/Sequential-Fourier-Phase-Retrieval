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

data_1 = load('person01_boxing_d2_uncomp_frames.mat');
data_2 = load('person01_handwaving_d2_uncomp_frames.mat');
data_3 = load('person10_handclapping_d2_uncomp_frames.mat');

num_frames = 10;
num_seq = 3;
images_1 = data_1.video(6:2:2*(num_frames-1)+6,:,:,:);
images_2 = data_2.video(4:2:2*(num_frames-1)+4,:,:,:);
images_3 = data_3.video(117:2:2*(num_frames-1)+117,:,:,:);

border_sizes = [5 10 15 20 25 30;5 10 15 15 15 15];

psnr_all = zeros(size(border_sizes,2),num_seq,num_frames);
psnr_ssim = zeros(size(border_sizes,2),num_seq,num_frames);


for p = 1:size(border_sizes,2)
    
    clc,p,tic
    
    
    for d = 1:num_seq
        %
        for im = 1: num_frames
            
            switch d
                case 1
                    image_org = im2double(images_1(im,:,:,:));
                case 2
                    image_org = im2double(images_2(im,:,:,:));
                case 3
                    image_org = im2double(images_3(im,:,:,:));
            end
            
            image_org = rgb2gray(squeeze(image_org));
            I_mod = image_org;
            
            m = size(image_org,1);
            n = size(image_org,2);
            
            % Design the border
            known_border_width_columns = border_sizes(1,p);
            known_border_width_rows = border_sizes(2,p);
            
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
                
                I_est_psnr = psnr(I_est,I_mod);
                I_est_ssim = ssim(I_est,I_mod);
                
                I_est_psnr_unknown = psnr(I_est(unknownRows,unknownColumns),I_mod(unknownRows,unknownColumns));
                I_est_ssim_unknown = ssim(I_est(unknownRows,unknownColumns),I_mod(unknownRows,unknownColumns));
                
                %                 axes(ha(im))
                %                 imagesc(I_mod)
                %                 if (im == 1)
                %                     ylabel('Original','FontSize',34)
                %                 end
                %
                %                 set(gca,'xtick',[])
                %                 set(gca,'ytick',[])
                %
                %                 axes(ha(im+num_frames))
                %                 imagesc(I_est);
                %                 if (im == 1)
                %                     ylabel('Recovered','FontSize',34)
                %                 end
                %
                %                 if (im == 1)
                %                     title(sprintf('(%0.2f, %0.2f)',I_est_psnr,I_est_ssim),'FontSize',22)
                %                 else
                %                     title(sprintf('(%0.2f, %0.2f)',I_est_psnr,I_est_ssim),'FontSize',22)
                %                 end
                %
                %                 colormap gray
                %                 set(gca,'xtick',[])
                %                 set(gca,'ytick',[])
                %
            end
            
            psnr_all(p,d,im) = I_est_psnr;
            ssim_all(p,d,im) = I_est_ssim;
            
            psnr_all_unknown(p,d,im) = I_est_psnr_unknown;
            ssim_all_unknown(p,d,im) = I_est_ssim_unknown;
            
        end
    end
    
end

%% PSNR and SSIM

known_pixels_percentage = (size(images_1,2)* size(images_1,3)- (size(images_1,2)- 2*border_sizes(1,:)).*(size(images_1,3)-2*border_sizes(2,:)))/size(images_1,2)/size(images_1,3)*100;

figure               
plot(known_pixels_percentage,mean(squeeze(psnr_all_unknown(:,1,:)),2),'r-*','Linewidth',2.5)
hold all
plot(known_pixels_percentage,mean(squeeze(psnr_all_unknown(:,2,:)),2),'g-o','Linewidth',2.5)
hold all
plot(known_pixels_percentage,mean(squeeze(psnr_all_unknown(:,3,:)),2),'b-^','Linewidth',2.5)
grid on
grid minor
xlabel('Known Area (%)','FontSize',18)
ylabel('Average PSNR (dB)','FontSize',18)

legend('Sequence 1','Sequence 2','Sequence 3','FontSize',20)
ax = gca ;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;


figure
plot(known_pixels_percentage,mean(squeeze(ssim_all_unknown(:,1,:)),2),'r-*','Linewidth',2.5)
hold all
plot(known_pixels_percentage,mean(squeeze(ssim_all_unknown(:,2,:)),2),'g-o','Linewidth',2.5)
hold all
plot(known_pixels_percentage,mean(squeeze(ssim_all_unknown(:,3,:)),2),'b-^','Linewidth',2.5)
grid on
grid minor
xlabel('Known Area (%)','FontSize',18)
ylabel('Average SSIM','FontSize',18)

legend('Sequence 1','Sequence 2','Sequence 3','FontSize',20)

ax = gca ;
ax.YAxis.FontSize = 20;
ax.XAxis.FontSize = 20;




