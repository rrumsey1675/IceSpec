function sfrcalc(varargin)
% Calculate SFR or MTF from an Excel-formatted text data file from ImageJ.
% The frequency scale is assumed to be logarithmic.
% Command line input data [default]:
% filename. minimum frequency (lp/mm) [2], maximum frequency (lp/mm) [200],
% Picture Height [24 mm]; optional
% Example:  sfrcalc inputfile.tif 2 80 15.1
%           EOS-10D 70-200 f/4 @ 70 mm, f/8

inpfile = varargin{1};  % data file from ImageJ (convert from cell.)
fileprt = strrep(inpfile,'_','\_');  % To print correctly in plots.
minlp = 2;  maxlp = 200;  picht = 24;  % Defaults for min, max lp/mm and picture height.

if (nargin>=3)
   minlp = str2num(varargin{2});  maxlp = str2num(varargin{3});  % Scale lp/mm min, max.
end

if (nargin>=4)
   picht = str2num(varargin{4});  % Picture height mm.
end

descr = input('Enter system description: ','s');
[fid,MESSAGE] = fopen(inpfile, 'rt');  % open data file for read
if (fid<0) error(MESSAGE);  end
[arrinp,l_ar] = fscanf(fid, '%f %f');
fclose(fid);  close all;
fprintf('%d data points read in\n',l_ar);

% Data in the first column is unreliable-- limited precision. Throw it out.
arr2 = arrinp(2:2:l_ar);  l_ar = length(arr2);
dif1 = diff(arr2);
fprintf('%d arr2 \n',arr2);
fprintf('%d data points read in\n',l_ar);
cmap = zeros(256,3);  % Color map: gray scale. Apparently gamma corrected.
gamma = 1;  cmap(1:256,1) = ([0:255]'/255).^(1/gamma);
cmap(1:256,2) = cmap(1:256,1);  cmap(1:256,3) = cmap(1:256,1);
yim = ones(25,1)*arr2';  % Input image for check.
yim(1,1:l_ar) = 0;  yim(1:25,1) = 0;  yim(1:25,l_ar) = 0;   % Black line on top, sides.

%         Detect peaks.
d1 = (dif1>0);  % 1 or 0.
d1 = (d1(2:l_ar-1) ~= d1(1:l_ar-2));  % 1 at peaks; 0 elsewhere.
% d2 = d1.*(abs(arrflt(1:l_ar-1))>thresh*maxflt);  % Detection threshold
% peak_loc = find(d2==1);  % integer peak locations.
peak_loc = find(d1==1)+1;  % integer peak locations.
n_peaks = length(peak_loc);  % number of detected peaks.
fprintf('%d detected peaks\n',n_peaks);
peak_ampl = arr2(peak_loc);  % End peak detection.

if (nargin<3)
   figure;  plot(peak_loc, peak_ampl);  grid on;  zoom on;  title(descr);  axis tight;
   p1 = get(gcf,'Position');  p1(4) = p1(4)-60; p1(3) = p1(3)-80;
   set(gcf,'Position',p1);
   fprintf('Scale values not entered. End program.\n');
else
   summ = peak_ampl(1:n_peaks-1)+peak_ampl(2:n_peaks);
   resp1 = abs(diff(peak_ampl))./summ;  resp1 = resp1/max(resp1);
   % Use equations from www.normankoren.com/Tutorials/MTF5.html
   lpratio = maxlp/minlp;
   xax1 = (0:l_ar-1)/(l_ar-1);  % Input data x-axis.
   lpmm1 = minlp*lpratio.^xax1;
   figure;  p1 = get(gcf,'Position');  p1(4) = p1(4)-60; p1(3) = p1(3)-80;
   plot(lpmm1,arr2);  grid on;  zoom on;  title(descr);  set(gcf,'Position',p1);
   xlabel('Line pairs per millimeter (lp/mm)');  ylabel('Input data');
   xax2 = peak_loc(2:n_peaks)/(l_ar-1);  % X-axis for MTF;
   lpmm2 = minlp*lpratio.^xax2;
   
   figure; p1 = get(gcf,'Position');  % Composite figure.
   p1(2) = p1(2)-120;   p1(4) = p1(4)+80; p1(3) = p1(3)-140;  set(gcf,'Position',p1);
   % p2 = get(gcf,'PaperPosition');
   % p2(2) = p2(2)-1;  p2(4) = p2(4)+2;  set(gcf,'PaperPosition',p2);
   subplot('Position',[.11 .92 .85 .03]);
   image(yim);  colormap(cmap);  axis off;  title(descr);
   subplot('Position',[.11 .63 .85 .29]); 
   plot(arr2);  grid on;  zoom on;  axis tight;
   xlabel(['Log scale from ' num2str(minlp) ' to ' num2str(maxlp) ' lp/mm']);
   ylabel('Input data');  grid on; zoom on;
   ax1 = axis;  a1 = ax1(2)-ax1(1);  a2 = ax1(4)-ax1(3);
   text(ax1(1)+.4*a1,ax1(3)+.07*a2,fileprt);
   subplot('Position',[.11 .08 .85 .46]);  % subplot(2,1,2);
   plot(lpmm2,resp1);  grid on;  zoom on;
   xlabel('Line pairs per millimeter (lp/mm)');  ylabel('Spatial frequency response');
   ax1 = axis;  a1 = ax1(2)-ax1(1);  a2 = ax1(4)-ax1(3);
   if (nargin>=4)  % Do plot in Line widths/Picture height = LW/PH.
      text(ax1(1)+.35*a1,ax1(3)+.92*a2,['To obtain Line widths per picture height']);
      text(ax1(1)+.35*a1,ax1(3)+.84*a2,['(lw/ph), where ph = ' num2str(picht) ' mm,']);
      text(ax1(1)+.35*a1,ax1(3)+.76*a2,['multiply by ' num2str(2*picht)]);
      figure;  set(gcf,'Position',p1);  % New plot of LW/PH.
      subplot('Position',[.11 .92 .85 .03]);
      image(yim);  colormap(cmap);  axis off;  title(descr);
      subplot('Position',[.11 .63 .85 .29]); 
      plot(arr2);  grid on;  zoom on;  axis tight;
      xlabel(['Log scale from ' num2str(minlp) ' to ' num2str(maxlp) ' lp/mm']);
      ylabel('Input data');  grid on; zoom on;
      ax1 = axis;  a1 = ax1(2)-ax1(1);  a2 = ax1(4)-ax1(3);
      text(ax1(1)+.4*a1,ax1(3)+.07*a2,fileprt);
      subplot('Position',[.11 .08 .85 .46]);  % subplot(2,1,2);
      plot(lpmm2*2*picht,resp1);  grid on;  zoom on;
      xlabel(['Line widths per picture height (lw/ph) for ph = ' num2str(picht) ' mm']);
      ylabel('Spatial frequency response');
      ax1 = axis;  a1 = ax1(2)-ax1(1);  a2 = ax1(4)-ax1(3); 
      text(ax1(1)+.42*a1,ax1(3)+.85*a2,['To obtain lp/mm, divide by ' num2str(2*picht)]);
   end
end