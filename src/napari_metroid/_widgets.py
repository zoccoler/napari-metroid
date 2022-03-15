"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from qtpy import QtWidgets, QtCore
import numpy as np

'''#########################################################################
                        Modified FunctionGui widget
   #########################################################################'''
from magicgui.widgets import FunctionGui
from ._bssd import get_noise_power, wavelet_denoise, get_signal_power

def get_components(ROIs_means_corrected,time,inactive_msk,t_sig_onset,
            method = 'ICA',n_comp = 2,wavelet = 'Haar',whiten = True):
    if t_sig_onset==0:
        t_sig_onset = None
    if n_comp==1:
        n_comp=2

    if (method=='ICA') | (method=='wICA'):
        from sklearn.decomposition import FastICA
        bss = FastICA(n_components=n_comp,max_iter=2000,tol=0.01)
    if (method=='PCA') | (method=='wPCA'):
        from sklearn.decomposition import PCA
        bss = PCA(n_components=n_comp,whiten=whiten)

    components = bss.fit_transform(ROIs_means_corrected)  # Estimate sources

    return(components, bss)

def denoise_manual(ROIs_means_corrected, time, inactive_msk, method, wavelet,
                   components, selected_source_idx, bss):
    components_filt = np.zeros_like(components)
    components_filt[:,selected_source_idx] = components[:,selected_source_idx]
    selected_source_idx = np.atleast_1d(selected_source_idx)

    if (method=='wPCA') | (method=='wICA'):
        for i in range(np.size(selected_source_idx)):
            components_filt[:,selected_source_idx[i]] = wavelet_denoise(
                components_filt[:, selected_source_idx[i]], time, wave=wavelet
                )
    ROIs_means_filtered = bss.inverse_transform(components_filt)
    ROIs_means_filtered = ROIs_means_filtered - np.median(
        ROIs_means_filtered[inactive_msk, :], axis=0
        )

    noise_power = get_noise_power(ROIs_means_corrected, time,
                                  inactive_msk=inactive_msk)
    signal_power = get_signal_power(ROIs_means_filtered, time, inactive_msk)

    SNR = signal_power / noise_power
    SNR_dB = np.zeros_like(SNR)
    SNR_dB[SNR>0] = 10 * np.log10(SNR[SNR > 0])

    return(ROIs_means_filtered, SNR_dB)

# Dock widget class that comprehends Get_signals, Remove_photobleaching and BSSD
class Ui_dock_widget(FunctionGui):
    def __init__(self, parent, napari_viewer, function, param_options={}):
        self.param_options = param_options
        super().__init__(
          function,
          call_button=True,
          layout='vertical',
          param_options=self.param_options
        )
        self.main_widget = parent
        self.function = function
        self.viewer = napari_viewer
        self.ROIs_avg, self.time_array, self.ROIs_avg_corrected = [],[],[]

    # Overrides the default call function to add a canvas dock widget when a
    #  label in a label layer is doubleclicked
    def __call__(self):
        # Regular call to function, getting outputs here (which are not new layers)
        if self.function.__name__ == 'get_ROIs_average_over_time':
            self.ROIs_avg,self.time_array = super().__call__()
            self.main_widget.outputs.raw_signals = self.ROIs_avg
            self.main_widget.outputs.time = self.time_array
            y_data_to_plot = [self.ROIs_avg]
        elif self.function.__name__ == 'photob_remove':
            if len(self.main_widget.outputs.raw_signals)>0:
                self.ROIs_avg = self.main_widget.outputs.raw_signals
                self.time_array = self.main_widget.outputs.time
            self.ROIs_avg_corrected, self.inactive_msk_array, self.onset, \
                self.end = super().__call__(
                    ROIs_means = self.ROIs_avg, time = self.time_array
                    )
            self.main_widget.outputs.corrected = self.ROIs_avg_corrected
            self.main_widget.outputs.inactive_msk = self.inactive_msk_array
            self.main_widget.outputs.t_sig_onset = self.onset
            self.main_widget.outputs.t_sig_end = self.end
            y_data_to_plot = [self.ROIs_avg, self.ROIs_avg_corrected]
        elif self.function.__name__ == 'denoise':
            if len(self.main_widget.outputs.raw_signals)>0:
                self.ROIs_avg = self.main_widget.outputs.raw_signals
                if len(self.main_widget.outputs.corrected)>0:
                    self.ROIs_avg_corrected = self.main_widget.outputs.corrected
                else:
                    self.ROIs_avg_corrected = self.main_widget.outputs.raw_signals
                self.time_array = self.main_widget.outputs.time
                self.onset = self.main_widget.outputs.t_sig_onset
                if len(self.main_widget.outputs.inactive_msk)>0:
                    self.inactive_msk_array = self.main_widget.outputs.inactive_msk
                else:
                    self.inactive_msk_array = None
            # If automatic choice, call regular denoise function
            if self.main_widget.bssd_widget.autoselect.value=='auto':
                self.ROIs_avg_filtered, \
                    self.components, \
                        self.selected_source_idx, \
                            self.SNR_dB = super().__call__(
                                ROIs_means_corrected = self.ROIs_avg_corrected,
                                time = self.time_array,
                                inactive_msk = self.inactive_msk_array,
                                t_sig_onset = self.onset)
                self.main_widget.outputs.filtered = self.ROIs_avg_filtered
                self.main_widget.outputs.components = self.components
                self.main_widget.outputs.selected_source_idx \
                    = self.selected_source_idx
                self.main_widget.outputs.SNR_dB = self.SNR_dB
                y_data_to_plot = [self.ROIs_avg,
                                  self.ROIs_avg_corrected,
                                  self.ROIs_avg_filtered]
            # if manual selection required:
            #   - get components
            #   - plot components in a new canvas widget and connect event to
            #       allow user select sources by clicking on chosen axes
            else:
                self.main_widget.outputs.components, \
                    self.bss = get_components(
                        ROIs_means_corrected = self.ROIs_avg_corrected,
                        time = self.time_array,
                        inactive_msk = self.inactive_msk_array,
                        t_sig_onset = self.onset,
                        method = self.main_widget.bssd_widget.method.value,
                        n_comp = self.main_widget.bssd_widget.n_comp.value,
                        wavelet = self.main_widget.bssd_widget.wavelet.value,
                        whiten = self.main_widget.bssd_widget.whiten.value)
                self.main_widget.manual_sel_canvas_widget.canvas._add_axes(
                    self.main_widget.outputs.components.shape[1])
                # Plot components
                for i in range(self.main_widget.outputs.components.shape[1]):
                    self.main_widget.manual_sel_canvas_widget._update_plot(
                        self.main_widget.outputs.time,
                        self.main_widget.outputs.components[:,i],
                        i,'b',
                        t_sig_onset=self.main_widget.outputs.t_sig_onset,
                        inactive_msk=self.main_widget.outputs.inactive_msk)
                self.viewer.window.add_dock_widget(
                    self.main_widget.manual_sel_canvas_widget, area='bottom')

                # connect click on axes event to 'onclick' callback function
                self.cid = \
                    self.main_widget.manual_sel_canvas_widget.canvas.mpl_connect(
                        'button_press_event',
                        self.main_widget.manual_sel_canvas_widget.onclick)
        # If canvas widget was called, update it
        if self.main_widget.cdw_instance is not None:
            if self.main_widget.bssd_widget.autoselect.value != 'manual':
                self.main_widget.canvas_widget._update_plots_after_run(
                    self.time_array, y_data_to_plot)

'''#########################################################################
                                Plot widget
   #########################################################################'''
class My_Line:
    '''Custom line class to store line data when axes are re-created'''
    def __init__(self,x,y,color):
        self.x = x
        self.y = y
        self.color = color
class My_Axes(My_Line):
    '''Custom axes class to store axes info when axes are re-created'''
    def __init__(self):
        self.lines = []
    def _add_line(self,line):
        self.lines.append(line)

class MplCanvas(FigureCanvas):
    """
    Defines the canvas of the matplotlib window
    """
    def __init__(self):
        self.fig = Figure()                         # create figure
        self.previous_axes_list = []
        self._add_axes(1)
        FigureCanvas.__init__(self, self.fig)       # initialize canvas
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,
                                    QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.previous_axes_list = []

    def _match_napari_layout(self, idx,color='white'):
        self.fig.set_facecolor('#00000000')
        # changing color of plot background to napari main window color
        self.fig.axes[idx].set_facecolor('#00000000')

        # changing colors of all axis
        self.fig.axes[idx].spines['bottom'].set_color(color)
        self.fig.axes[idx].spines['top'].set_color(color)
        self.fig.axes[idx].spines['right'].set_color(color)
        self.fig.axes[idx].spines['left'].set_color(color)

        self.fig.axes[idx].xaxis.label.set_color(color)
        self.fig.axes[idx].yaxis.label.set_color(color)

        # changing colors of axis labels
        self.fig.axes[idx].tick_params(axis='x', colors=color)
        self.fig.axes[idx].tick_params(axis='y', colors=color)

    def _add_axes(self,N):
        '''adds (or removes) axes and replots previous data'''
        if len(self.fig.axes)>0:
            self.previous_axes_list = []
            for ax in self.fig.axes:
                previous_axes = My_Axes()
                for line in ax.lines:
                    previous_line = My_Line(x = line.get_xdata(),
                                            y = line.get_ydata(),
                                            color = line.get_color())
                    previous_axes._add_line(previous_line)
                self.previous_axes_list.append(previous_axes)
                ax.remove()

        gs = self.fig.add_gridspec(N, 1,hspace=0)
        for i in range(N):
            ax1 = self.fig.add_subplot(gs[i])
            ax1.set_picker(True)
            try:
                for line in self.previous_axes_list[i].lines:
                    ax1.plot(line.x,line.y,color=line.color)
            except IndexError:
                pass
            self._match_napari_layout(i)


class MyNavigationToolbar(NavigationToolbar):
    """Custom Navigation toolbar to match napari style and custom save figure
    with transparent background"""
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.canvas = canvas
    def save_figure(self):
        # Override default save figure callback function behaviour
        # test saving figure with transparent backgorund
        self.canvas.fig.set_facecolor("#00000000")
        for i in range(len(self.canvas.fig.axes)):
            self.canvas.fig.axes[i].set_facecolor("#00000000")
            self.canvas._match_napari_layout(i,'black')
        super().save_figure()
        for i in range(len(self.canvas.fig.axes)):
            self.canvas._match_napari_layout(i)
        self.canvas.update()

class Canvas_Widget(QtWidgets.QWidget):
    """Widget for plotting signals over time.
    It is used either for ROI signals or for components when manual selection
    is chosen."""
    def __init__(self, parent, napari_viewer, src_selection=False):
        super().__init__()
        self.viewer = napari_viewer
        self.main_widget = parent
        self.src_selection = src_selection
        self.selected_axes_list = []
        self.plotted_signals_info = {'labels' : [], 'colors': []}
        self.setMinimumSize(QtCore.QSize(100,300))
        self.vboxLayout = QtWidgets.QVBoxLayout()
        self.canvas = MplCanvas()

        self.toolbar = MyNavigationToolbar(self.canvas, self)
        self.vboxLayout.addWidget(self.canvas)
        self.vboxLayout.addWidget(self.toolbar)
        # If widget is used to plot components/sources:
        #   - add a Done button that gets selected sources, calls
        #       'denoise_manual' to perform inverse transform with selected
        #       sources, and closes this canvas widget
        if self.src_selection==True:
            self.done_button = QtWidgets.QPushButton(self)
            self.done_button.setText("Done")
            self.vboxLayout.addWidget(self.done_button)
            self.done_button.clicked.connect(self.onclose)
        self.setLayout(self.vboxLayout)

    def _update_plot(self, time, y, canvas_idx, color,clicked_label=0,
                     t_sig_onset=None,inactive_msk=None):
        (line,) = self.canvas.fig.axes[canvas_idx].plot(time,y,color=color)
        # If widget is used to plot components/sources
        if self.src_selection==True:
            self.canvas.fig.axes[canvas_idx].axvline(t_sig_onset,color='k',
                                                     linestyle='--',
                                                     alpha=0.3)
            ymax = np.mean(y[inactive_msk]) + 2*np.std(y[inactive_msk])
            ymin = np.mean(y[inactive_msk]) - 2*np.std(y[inactive_msk])
            span = self.canvas.fig.axes[canvas_idx].axhspan(
                ymin=ymin, ymax=ymax, alpha=0.2, label='Noise CI95',
                color='white')
            ymin = 1.8*np.amin(y)
            ymax = 1.8*np.amax(y)
            self.canvas.fig.axes[canvas_idx].set_ylim(ymin, ymax)
            self.canvas.fig.axes[canvas_idx].legend(handles=[span],
                                                    loc='upper right')
        # If widget is used to plot ROI signals
        else:
            if clicked_label not in self.plotted_signals_info['labels']:
                self.plotted_signals_info['labels'].append(clicked_label)
                self.plotted_signals_info['colors'].append(color)
        self.canvas.fig.axes[canvas_idx].axis('on')
        if canvas_idx>0:
            self.canvas.fig.axes[canvas_idx-1].get_xaxis().set_visible(False)
        self.canvas.fig.tight_layout()

    def _update_plots_after_run(self,time,y):
        '''Everytime a Run button is clicked, readily update the plots'''
        for index, ax in enumerate(self.canvas.fig.axes):
            # If something was drawn, update it
            plotted_labels = self.plotted_signals_info['labels']
            plotted_colors = self.plotted_signals_info['colors']
            if len(plotted_labels)>0:
                ax.clear()
                for label, color in zip(plotted_labels, plotted_colors):
                    self._update_plot(time, y[index][:,label-1], index,
                                      color,label)
                    self.canvas.fig.canvas.draw_idle()
    def _clear(self):
        '''Clear all plots'''
        for i in range(len(self.canvas.fig.axes)):
            self.canvas.fig.axes[i].clear()
            self.plotted_signals_info['labels'] = []
            self.plotted_signals_info['colors'] = []

    # Modified from https://stackoverflow.com/a/39351847/11885372
    def onclick(self, event):
        '''Get which axes were clicked and highlights them in white'''
        for i, ax in enumerate(self.canvas.fig.axes):
            if ax == event.inaxes:
                if i not in self.selected_axes_list:
                    self.selected_axes_list.append(i)
                    event.inaxes.patch.set_facecolor('white')
                    event.canvas.draw()
                else:
                    self.selected_axes_list.pop(self.selected_axes_list.index(i))
                    event.inaxes.patch.set_facecolor('#00000000')
                    event.canvas.draw()
    def onclose(self,event):
        # Get manually selected source(s)
        self.main_widget.outputs.selected_source_idx = self.selected_axes_list
        # Remove noise by rebuilding signals only with selected sources
        self.main_widget.outputs.filtered, \
            self.main_widget.outputs.SNR_dB = denoise_manual(
                self.main_widget.outputs.corrected,
                self.main_widget.outputs.time,
                self.main_widget.outputs.inactive_msk,
                self.main_widget.bssd_widget.method.value,
                self.main_widget.bssd_widget.wavelet.value,
                self.main_widget.outputs.components,
                self.selected_axes_list,
                self.main_widget.bssd_widget.bss)
        # Updated data to be plotted
        y_data_to_plot = [self.main_widget.outputs.raw_signals,
                          self.main_widget.outputs.corrected,
                          self.main_widget.outputs.filtered]
        # Plot new data
        self.main_widget.canvas_widget._update_plots_after_run(
            self.main_widget.outputs.time, y_data_to_plot)
        # Disconnect source selection click events
        self.canvas.mpl_disconnect(self.main_widget.bssd_widget.cid)
        # Remove source selection from interface
        self._clear()
        # Remove canvas widget from napari
        self.viewer.window.remove_dock_widget(self)

