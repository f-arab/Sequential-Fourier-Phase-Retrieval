%% Sequential Recovery for Solving Fourier Phase Retrieval
% Suppose a known patch is added to the right side of the image
% Size of original image: m*n
% Size of the known patch: m*p
% Known patch: pinhole (we can move the pinhole to discuss the separation condition!?)

%%
clear variables
clc
close all

%% Load images

% images = {'baboon.png','cameraman.tif','coins.png'};
images = {'baboon.png'};

pinhole_pos = [16];
psnr_all = zeros(length(pinhole_pos),length(images));
psnr_ssim = zeros(length(pinhole_pos),length(images));

fig = figure(300); fig.Position = [100 100 300 320];
[ha, pos] = tight_subplot(1,1, [0.08 0.005], [0.17 0.01], [0.005 0.005]);

for p = 1:length(pinhole_pos)
    clc,p,tic
    
    for im = 1: length(images)
        image_name = images{im};
        image_org = im2double(imread(image_name));
        image_org = imresize(image_org,[64 64]);
        
        for c = 1:size(image_org,3)
            image_org_c = image_org(:,:,c);
            m_org = size(image_org_c,1);
            n_org = size(image_org_c,2);
            
            % known patch
            patch = zeros(m_org,pinhole_pos(p));
            patch(1,end) = 64;
            
            I_mod = [image_org_c patch];
            
            c_n = I_mod(:,end);
            [m,n] = size(I_mod);
            
            R_corr = xcorr2(I_mod,I_mod);
            
            toeplitz_left = @(z) fliplr(toeplitz([z; zeros(m-1,1)], [z(1),zeros(1,m-1)]));
            toeplitz_right = @(z) toeplitz([flipud(z); zeros(m-1,1)], [z(end),zeros(1,m-1)]);
            
            c_n_T = toeplitz_right(c_n);
            c_1 = pinv(c_n_T)*R_corr(:,1);
            c_1_T = toeplitz_left(c_1);
            
            c_1_p = pinv(c_n_T)*R_corr(:,1:pinhole_pos(p));
            
            %% Sequential Recover
            knownRows = [];
            unknownRows = setdiff([1:m],knownRows);
            H_c = [c_1_T(:,unknownRows) c_n_T(:,unknownRows)];
            H_cond = cond(H_c)
            H_c_inv = pinv(H_c);
            
            knownColumns = [n-pinhole_pos(p)+1 : n];
            
            % Recovery
            I_est = 0*I_mod;
            I_est(:,knownColumns) = I_mod(:,knownColumns);         % set known columns: first and last ones
            I_est(knownRows,:) = I_mod(knownRows,:);               % set known rows at each column: first and last pixels
            I_est(:,1:pinhole_pos(p)) = c_1_p;
            
            
            %         axes(ha(im ))
            %         imagesc(log10(R_corr.*R_corr))
            %         if (im == 1)
            %             ylabel('Autocorrelation','FontSize',18)
            %         end
            %
            %         set(gca,'xtick',[])
            %         set(gca,'ytick',[])
            %
            
            for k = pinhole_pos(p)+1:ceil(size(I_est,2)/2)
                
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
            I_est_colored(:,:,c)= I_est;
        end
        I_est_psnr = psnr(I_est_colored(1:m_org,1:n_org,:),image_org);
        I_est_ssim = ssim(I_est_colored(1:m_org,1:n_org,:),image_org);
        
        axes(ha(1))
        imagesc(I_est_colored(1:m_org,1:n_org,:));
        xlabel(sprintf('(%0.2f, %0.2f)',I_est_psnr,I_est_ssim),'FontSize',28)
        
        colormap gray
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        drawnow
        
    end
end

%% PSNR and SSIM


% figure
% plot(pinhole_pos,psnr_all(:,1),'-*','Linewidth',2.5)
% hold all
% plot(pinhole_pos,psnr_all(:,2),'-o','Linewidth',2.5)
% hold all
% plot(pinhole_pos,psnr_all(:,3),'-^','Linewidth',2.5)
% grid on
% grid minor
% xlabel('Separation (p)','FontSize',18)
% ylabel('PSNR (dB)','FontSize',18)
%
% legend('Baboon','Cameraman','Coins','FontSize',20)
% ax = gca ;
% ax.YAxis.FontSize = 20;
% ax.XAxis.FontSize = 20;
%
%
% figure
% plot(pinhole_pos,ssim_all(:,1),'-*','Linewidth',2.5)
% hold all
% plot(pinhole_pos,ssim_all(:,2),'-o','Linewidth',2.5)
% hold all
% plot(pinhole_pos,ssim_all(:,3),'-^','Linewidth',2.5)
% grid on
% grid minor
% xlabel('Separation (p)','FontSize',18)
% ylabel('SSIM','FontSize',18)
%
% legend('Baboon','Cameraman','Coins','FontSize',20)
%
% ax = gca ;
% ax.YAxis.FontSize = 20;
% ax.XAxis.FontSize = 20;
