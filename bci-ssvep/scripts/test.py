# import matplotlib
# matplotlib.use('TkAgg')  # явно указываем бэкенд
# import matplotlib.pyplot as plt

# import mne

# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_raw_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

# raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
# raw.pick('eeg')

# raw.plot(duration=5, n_channels=10, title='Raw EEG Signal')
# plt.show(block=True)  # держит окно открытым

# import mne
# import matplotlib.pyplot as plt

# sample_data_folder = mne.datasets.sample.data_path()
# raw = mne.io.read_raw_fif(
#     sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif',
#     preload=True
# ).pick('eeg')

# raw.filter(1, 40)

# # Полный диапазон — topomap сам разобьёт на Delta, Theta, Alpha, Beta
# raw.compute_psd(fmin=1, fmax=40).plot_topomap()
# plt.show(block=True)

# import mne
# import matplotlib.pyplot as plt

# fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
# subjects_dir = fs_dir.parent

# sample_data_folder = mne.datasets.sample.data_path()

# # Загружаем ВСЕ каналы сначала
# raw = mne.io.read_raw_fif(
#     sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif',
#     preload=True
# )

# raw.filter(1, 40)

# # Находим события ДО того как выбрать только EEG
# events = mne.find_events(raw, stim_channel='STI 014')

# # Теперь оставляем только EEG
# raw.pick('eeg')
# raw.set_eeg_reference(projection=True)

# epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5,
#                     baseline=(None, 0), preload=True)
# evoked = epochs.average()

# fwd = mne.make_forward_solution(
#     evoked.info,
#     trans='fsaverage',
#     src=str(fs_dir / 'bem' / 'fsaverage-ico-5-src.fif'),
#     bem=str(fs_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'),
#     eeg=True, meg=False, verbose=True
# )

# noise_cov = mne.compute_covariance(epochs, tmax=0)

# inverse_op = mne.minimum_norm.make_inverse_operator(
#     evoked.info, fwd, noise_cov, verbose=True
# )
# stc = mne.minimum_norm.apply_inverse(
#     evoked, inverse_op, lambda2=1./9., method='dSPM'
# )

# brain = stc.plot(
#     subject='fsaverage',
#     subjects_dir=subjects_dir,
#     hemi='both',
#     time_viewer=True,
# )
# plt.show(block=True)

# brain.show()
# import time
# time.sleep(60)  # держит окно открытым 60 секунд

# Стимул должен быть ТОЧНЫМ делителем refresh rate монитора
# Если монитор 60 Гц — хорошие частоты: 6, 7.5, 10, 12, 15, 20 Гц
# Если монитор 144 Гц — больше вариантов: 8, 9, 12, 18 Гц и т.д.

# Проверить в Python:
from psychopy import visual, core
import numpy as np

# Создаём окно — fullscr важно для точного timing
win = visual.Window(
    size=[1280, 720],
    fullscr=False,  # поставь True для реального эксперимента
    color='gray',
    units='norm',
    allowGUI=False
)

# Два мигающих стимула
stim_left = visual.Rect(
    win, width=0.3, height=0.4,
    pos=(-0.5, 0),
    fillColor='white'
)
stim_right = visual.Rect(
    win, width=0.3, height=0.4,
    pos=(0.5, 0),
    fillColor='white'
)

# Фиксационный крест
fixation = visual.TextStim(win, text='+', color='black', height=0.1)

# Частоты стимулов (при 60 Гц мониторе)
freq_left = 10   # Гц — каждые 6 кадров
freq_right = 15  # Гц — каждые 4 кадра

fps = 60
frame = 0
clock = core.Clock()

# Запуск — 10 секунд
while clock.getTime() < 10.0:
    # Вычисляем нужно ли показывать стимул в этом кадре
    # Синусоида: если sin > 0 — белый, если < 0 — чёрный
    t = frame / fps

    left_on = np.sin(2 * np.pi * freq_left * t) > 0
    right_on = np.sin(2 * np.pi * freq_right * t) > 0

    stim_left.fillColor = 'white' if left_on else 'black'
    stim_right.fillColor = 'white' if right_on else 'black'

    stim_left.draw()
    stim_right.draw()
    fixation.draw()

    win.flip()  # синхронизация с VSync — ключевой момент
    frame += 1

win.close()
core.quit()