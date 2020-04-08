mat = load("/Users/yazhuo/Projects/ML project/seizure-detection/Volumes/Seagate/seizure_detection/competition_data/clips/Dog_1/dog_1_ictal_segment_1.mat");
data = mat.data;
data = data';
Fs = mat.freq;
newYlabels = {'channel1','channel2','channel3','channel4','channel5','channel6','channel7','channel8',...
               'channel9','channel10','channel11','channel12','channel13','channel14','channel15','channel16'};

%disp(data(:,1));
channel1 = data(:,1);    
trange = (0:(size(data,1)-1))/Fs; 
figure(1)
plot(trange, channel1);
title('Noisy time domain signal (dog1 ictal segment1)');
xlabel('Time (s)')

y1 = fft(channel1);
L = numel(trange);
Py1 = y1.*conj(y1)/(L/2);
f = 400/(L/2)*(0:L/4-1);
figure(2)
plot(f,Py1(1:L/4));

title('Power spectral density (dog1 ictal segment1)')
xlabel('Frequency (Hz)')

%ps1 = abs(y1.^2);
%plot(frange, ps1);
%title('power spectrum using FFT(channel 9)')
%xlabel('frequency')
%disp(trange);


%stackedplot(data, 'Title','EEG signals','DisplayLabels',newYlabels);
%xlabel('time');

%Y = fft(data); %computes the Fast Fourier Transform of your EEG data 
%ps1=abs(Y).^2; %computes the power spectrum
%stackedplot(ps1, 'Title','POWER SPECTRUM USING FFT METHOD','DisplayLabels',newYlabels);
%xlabel('frequency');

