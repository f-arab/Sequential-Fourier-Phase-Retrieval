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
% 'logo.tif'
image = 'cameraman.tif';
fig = figure(300); fig.Position = [100 250 400 400];
[ha, pos] = tight_subplot(2,1, [0.01 0.01], [0.01 0.01], [0.01 0.005])

image_name = image;
image_org = im2double(imread(image_name));
image_org = imresize(image_org,[64 64]);

m_org = size(image_org,1);
n_org = size(image_org,2);

p = [64, 16];
for pos = 1:2
% known patch
patch = zeros(m_org,n_org);
patch(1,p(pos) ) = 64;

I_mod = [image_org patch];
R_corr = xcorr2(I_mod,I_mod);

% I_mod_visual = [image_org normalize(patch)];
% axes(ha(pos))
% imagesc(I_mod_visual)

axes(ha(pos))
imagesc((R_corr.*sign(R_corr)).^(1/4))

% if (pos == 1)
% ylabel('(a)','FontSize',26)
% % title('Autocorrelation','FontSize',26)
% else 
% ylabel('(b)','FontSize',26)
% end
if (pos == 1)
    title('Autocorrelation','FontSize',26)
end

set(gca,'xtick',[])
set(gca,'ytick',[])
colormap gray

end
