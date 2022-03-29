"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from ._auto_mask import create_cell_mask
from ._mess import segment
from ._get_signals import get_ROIs_average_over_time
from ._remove_bleaching import photob_remove
from ._bssd import denoise
from magicgui import magicgui
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow
from ._widgets import Ui_dock_widget, Canvas_Widget
from qtpy import uic
from pathlib import Path

import napari

'''#########################################################################
                                Main Interface
   #########################################################################'''
class MainInterface(QMainWindow):
    outputs = {'time' : [], 'raw_signals' : [], 'corrected' : [],
               'filtered' : [], 'inactive_msk' : [], 'components' : [],
               'SNR_dB' : [], 't_sig_onset' : [], 't_sig_end' : [],
               'selected_source_idx' : []
               }

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        print('Parent path = ', str(Path(__file__).parent))
        self.UI_FILE = str(Path(__file__).parent / "ui//metroid_main_widget.ui")  # path to .ui file
        self.main_gui = uic.loadUi(self.UI_FILE, self)           # load QtDesigner .ui file

        self.outputs = DotDict(self.outputs)

        self.viewer = napari_viewer
        self.dock_widget_flag = False

        self.create_mask_widget =magicgui(create_cell_mask,auto_call=True)
        self.create_mask_widget._widget._layout.setAlignment(Qt.AlignTop)
        self.mess_widget = magicgui(segment,
                                    auto_call=True,
                                    n_ROIs_in={'label': 'Number of inner ROIs:',
                                                        'min': 4, 'step': 2},
                                    n_ROIs_out={'label': 'Number of outer ROIs:',
                                                        'min': 2})
        self.mess_widget._widget._layout.setAlignment(Qt.AlignTop)

        @self.create_mask_widget.called.connect
        @self.mess_widget.called.connect
        def clear_outputs():
            # Clear outputs when new mask or new ROIs are generated
            # Code from https://stackoverflow.com/a/22991990/11885372
            self.outputs = self.outputs.fromkeys(self.outputs, [])


        self.get_signals_widget = Ui_dock_widget(self,self.viewer,get_ROIs_average_over_time,
                                                 param_options={'frame_rate_info':
                                                                {'visible':True},
                                                                'frame_rate':
                                                                    {'label':'Frame Rate (frames/s):',
                                                                     'visible': True},
                                                                'label_image':
                                                                    {'choices' : []}})

        @self.get_signals_widget.frame_rate_info.changed.connect
        def show_frame_rate():
            if self.get_signals_widget.frame_rate.visible == False:
                self.get_signals_widget.frame_rate.show()
            else:
                self.get_signals_widget.frame_rate.hide()



        self.remove_photob_widget = Ui_dock_widget(self,self.viewer,photob_remove,
                                                   param_options={'t_sig_onset':
                                                                {'label':'Signal starts at (s):'},
                                                                't_sig_end':
                                                                {'label':'Signal ends at (s):'},
                                                                'label_image':
                                                                    {'choices' : []}})

        @self.remove_photob_widget.transitory.changed.connect
        def show_times():
            if self.remove_photob_widget.transitory == True:
                self.remove_photob_widget.t_sig_end.show()
            else:
                self.remove_photob_widget.t_sig_end.hide()


        bssd_options = ['ICA', 'PCA', 'wICA', 'wPCA']
        wavelet_options = ['Haar','dmey']
        source_selection_options = [('Automatic','auto'), ('Manual','manual') ]
        self.bssd_widget = Ui_dock_widget(self,self.viewer,denoise,
                                                   param_options={'method':
                                                                      {'label':'Method:','choices':bssd_options},
                                                                  'n_comp':
                                                                      {'label':'Number of Components:','min':1},
                                                                  'wavelet':
                                                                      {'label':'Wavelet:','choices':wavelet_options},
                                                                   'autoselect':
                                                                       {'label':'Source Selection:','choices':source_selection_options}})
        self.canvas_widget = Canvas_Widget(self, self.viewer)
        self.canvas_widget.setObjectName('Canvas Widget')

        self.manual_sel_canvas_widget = Canvas_Widget(self, self.viewer, True)
        self.manual_sel_canvas_widget.setObjectName('Source Selection')

        # List of widgets and list of docked widgets
        self.widgets_list = [self.create_mask_widget, self.mess_widget,
                             self.get_signals_widget, self.remove_photob_widget,
                             self.bssd_widget]
        self.widgets_names = ['Create Mask', 'MESS: Segment shape',
                              'Get Signals (time series)',
                              'Remove photobleaching', 'BSSD: Denoise']
        # Associate each main interface button to _update_gui function
        n_buttons = self.main_gui.horizontalLayout.layout().count()
        for i in range(n_buttons):
            widget = self.main_gui.horizontalLayout.layout().itemAt(i).widget()
            if isinstance(widget,QtWidgets.QToolButton):
                widget.clicked.connect(self._update_dock_widgets)
        # instance to dock_widgets
        self.dw_instance = None
        # instance to canvas dock_widgets
        self.cdw_instance = None
        # reference for ROIs label layer
        self.ROIs_label_layer = None

        self.plots_active = False

    # Docks the widget relative to the clicked button
    def _update_dock_widgets(self):
        if self.dw_instance is not None:
            self.viewer.window.remove_dock_widget(self.dw_instance)
            if self.cdw_instance is not None:
                self.viewer.window.remove_dock_widget(self.cdw_instance)

        sender = self.sender()
        # Identify button by last character of button name (which is a number)
        self.selected_widget = self.widgets_list[int(sender.objectName()[-1])-1]
        self.number = int(sender.objectName()[-1])
        # When get_signals or remove_bleaching are selected, put resulting layer
        # from MESS (segment result) into label_image combobox
        # (exclude cell mask label layer from the options)
        if ((self.number == 3) | (self.number == 4)):
            layers_list = self.viewer.layers
            self.ROIs_label_layer = None
            for i,layer in enumerate(layers_list):
                if isinstance(layer, napari.layers.labels.labels.Labels):
                    if layer.name=='segment result':
                        self.ROIs_label_layer = layer
            params = self.selected_widget.param_options
            function = self.selected_widget.function
            if self.ROIs_label_layer is not None:
                # Set unique choice to label_image as ROIs_label_layer
                params['label_image']['choices'] = [self.ROIs_label_layer]
                # Reinitialize widget to update choices displayed
                # TO DO: find a better way of doing this instead of calling __init__
                # and reassignin the callback function
                self.selected_widget.__init__(self,self.viewer,function,params)
                if self.number==3:
                    @self.selected_widget.frame_rate_info.changed.connect
                    def show_frame_rate_again():
                        if self.selected_widget.frame_rate.visible == False:
                            self.selected_widget.frame_rate.show()
                        else:
                            self.selected_widget.frame_rate.hide()
                        self.selected_widget.update()
                else:
                    @self.selected_widget.transitory.changed.connect
                    def show_times():
                        if self.selected_widget.transitory == True:
                            self.selected_widget.t_sig_end.show()
                        else:
                            self.selected_widget.t_sig_end.hide()
        # Add selected dock_widget
        self.selected_widget._widget._layout.setAlignment(Qt.AlignTop)
        self.dw_instance = self.viewer.window.add_dock_widget(self.selected_widget,
                                                              name = self.widgets_names[self.number-1],
                                                              area='right')

        # Add canvas widget
        if self.cdw_instance is not None:
            self.cdw_instance = self.viewer.window.add_dock_widget(self.canvas_widget,
                                                                   area='right')

        if self.ROIs_label_layer is not None:
            # Add axes matching the number of plots (had to clear axes in order to maintain reference to the same object)
            if self.number<3:
                self.number = 3
            self.canvas_widget.canvas._add_axes(self.number-2)
            if self.plots_active == False:
                self.plots_active = True
                # connect a callback that updates the line plot when
                # the user clicks on the image
                # Code below based on https://github.com/napari/napari/blob/main/examples/mpl_plot.py
                #   and https://github.com/napari/napari/blob/main/examples/custom_mouse_functions.py
                @self.ROIs_label_layer.mouse_double_click_callbacks.append
                def plot_time_average(layer, event):
                    try:
                        if self.cdw_instance is not None:
                            self.viewer.window.remove_dock_widget(self.cdw_instance)
                        self.viewer.window.remove_dock_widget(self.dw_instance)
                        if 'alt' in event.modifiers: # Pressing ALT while clicking retains plots
                            pass
                        else:
                            self.canvas_widget._clear()

                        clicked_label = layer.data[int(event.position[1]),int(event.position[2])]
                        label_color = layer.get_color(clicked_label)
                        if clicked_label!=0:
                            if (self.number>=3) & (len(self.outputs.raw_signals)>0):
                                self.canvas_widget._update_plot(self.outputs.time, self.outputs.raw_signals[:,clicked_label-1],0,label_color,clicked_label)
                            if (self.number>=4) & (len(self.outputs.corrected)>0):
                                self.canvas_widget._update_plot(self.outputs.time, self.outputs.corrected[:,clicked_label-1],1,label_color,clicked_label)
                            if (self.number==5) & (len(self.outputs.filtered)>0):
                                self.canvas_widget._update_plot(self.outputs.time, self.outputs.filtered[:,clicked_label-1],2,label_color,clicked_label)
                        # Add selected dock_widget
                        self.dw_instance = self.viewer.window.add_dock_widget(self.selected_widget,
                                                                              name = self.widgets_names[self.number-1],
                                                                              area='right')
                        # Add canvas widget
                        self.cdw_instance = self.viewer.window.add_dock_widget(self.canvas_widget, area='right')
                    except IndexError:
                        pass

 # code based on https://stackoverflow.com/a/23689767/11885372
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
