import distinctipy
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QLoggingCategory
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QTreeWidget, QTreeWidgetItem, QHeaderView, QTabWidget, QLabel, QAbstractItemView
from rfrl_gym.renderers.ma_renderer import MultiAgentRenderer
import pdb
import time
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.fft import fftshift
import random
from rfrl_gym import repo_root_directory

class MultiAgentPyQtRenderer(MultiAgentRenderer, QMainWindow):
    def __init__(self, num_episodes, scenario_metadata, mode):
        super(MultiAgentPyQtRenderer, self).__init__(num_episodes, scenario_metadata)
        QLoggingCategory.setFilterRules('*=false')
        self.max_steps = scenario_metadata['environment']['max_steps']
        self.render_background = scenario_metadata['render']['render_background']
        self.mode = mode

        self.show_flag = 0
        self.win_width = 1400
        self.win_height = 900
        self.samples_per_step = 10000
        self.t = np.linspace(0, self.samples_per_step, self.samples_per_step)
        self.fc = np.linspace(-0.5, 0.5, self.num_channels+1)+1/self.num_channels/2
        self.sos = signal.butter(100, 1/self.num_channels, output='sos')

        # Initialize the main window size, color, and panels.
        self.setWindowTitle('RFRL GYM')
        self.setFixedWidth(self.win_width)
        self.setFixedHeight(self.win_height)
        if self.render_background == 'white':
            self.pen_color = 'k'
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            self.background = "background-color: " + str(self.render_background) + "; color: white;"
            self.setStyleSheet("QMainWindow {background: 'white';}")
        elif self.render_background == 'black':
            self.pen_color = 'w'
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')
            self.background = "background-color: " + str(self.render_background) + "; color: black;" 
            self.setStyleSheet("QMainWindow {background: 'black';}")
        self.main_panel = QGridLayout()
        
        # Initialize the main window widgets.
        self.__initialize_occupancy_view()
        self.__initialize_logo_view()
        self.__initialize_legend_view()     
        self.__initialize_cummulative_reward_view()
 
        if self.mode == 'abstract':
            self.main_panel.addWidget(self.occupancy_view,                           1, 0, 2, 2)
        elif self.mode == 'iq': # not yet fully implemented for multi-agent
            self.__initialize_spectrum_view()  
            self.__initialize_sensing_view()
            self.tabs1 = QTabWidget()
            self.tabs1.setStyleSheet("QTabWidget::pane { border: 0; }")
            self.tabs1.setTabPosition(QTabWidget.TabPosition.West)
            self.tabs1.addTab(self.spectrum_view, "Spectrum View")            
            self.tabs1.addTab(self.sensing_view, "Sensing View")
            self.tabs1.tabBar().setTabTextColor(0, QtGui.QColor(255,255,255))
            self.tabs1.tabBar().setTabTextColor(1, QtGui.QColor(255,255,255))
            self.tabs1.tabBar().setStyleSheet('QTabBar::tab{background: black; color: white; border-width: 1px; border-style: solid; border-color: white; border-bottom-color: white; border-top-left-radius: 6px; border-top-right-radius: 6px; border-bottom-left-radius: 6px; border-bottom-right-radius: 6px; min-height: 40px; padding: 2px; margin-bottom: 2px;}')

            self.main_panel.addWidget(self.tabs1,                                     1, 0, 2, 1)
            self.main_panel.addWidget(self.occupancy_view,                           1, 1, 2, 1)

        self.tabs2 = QTabWidget()
        self.tabs2.setStyleSheet("QTabWidget::pane { border: 0; }")
        self.tabs2.setTabPosition(QTabWidget.TabPosition.West)
        self.tabs2.addTab(self.cummulative_reward_view, "Step Rewards")            
        self.tabs2.tabBar().setTabTextColor(0, QtGui.QColor(255,255,255))
        self.tabs2.tabBar().setStyleSheet('QTabBar::tab{background: black; color: white; border-width: 1px; border-style: solid; border-color: white; border-bottom-color: white; border-top-left-radius: 6px; border-top-right-radius: 6px; border-bottom-left-radius: 6px; border-bottom-right-radius: 6px; min-height: 40px; padding: 2px; margin-bottom: 2px;}')

        self.label1 = QLabel()
        self.label1.setText('Data View')
        self.label1.setStyleSheet("QLabel { background-color : black; color : white; font-size: 11pt;}")
        self.label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label2 = QLabel()
        self.label2.setText('             Occupancy Map')
        self.label2.setStyleSheet("QLabel { background-color : black; color : white; font-size: 11pt;}")
        self.label2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label3 = QLabel()
        self.label3.setText('Occupancy Map Legend')
        self.label3.setStyleSheet("QLabel { background-color : black; color : white; font-size: 11pt;}")
        self.label3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.main_panel.addWidget(self.label1,                                        0, 0, 1, 1)
        self.main_panel.addWidget(self.label2,                                        0, 1, 1, 1)
        self.main_panel.addWidget(self.label3,                                        0, 2, 1, 1)
        self.main_panel.addWidget(self.legend_view,                                   1, 2, 2, 1)
        self.main_panel.addWidget(self.tabs2,                                         3, 0, 1, 2)
        self.main_panel.addWidget(self.logo_view,                                     3, 2, 1, 1)
        self.main_panel.setColumnStretch(0,16)
        self.main_panel.setColumnStretch(1,14)
        self.main_panel.setColumnStretch(2,6)
        if self.num_episodes != 1:
            self.__initialize_episode_reward_view()
            self.tabs2.addTab(self.episode_reward_view, "Episode Rewards")
            self.tabs2.tabBar().setTabTextColor(1, QtGui.QColor(255,255,255))

        self.main_panel.setRowStretch(0,1)
        self.main_panel.setRowStretch(1,12)
        self.main_panel.setRowStretch(2,12)
        self.main_panel.setRowStretch(3,12)   
        
        # Finalize the main window.
        self.widget = QWidget()
        self.widget.setLayout(self.main_panel)
        self.setCentralWidget(self.widget)

    def _render(self):

        # Clear and update the channel occupancy map.  
        self.__get_occupancy_image()     
        self.occupancy_image_item.setImage(self.occupancy_image)  

        if self.mode == 'iq':
            self.__get_spectrogram_image()
            self.__get_sensing_image()  
            self.spectrum_image_item.setImage(self.spectrum_image)
            self.sensing_image_item.setImage((self.sensing_image - np.min(self.sensing_image)) / \
                                             (np.max(self.sensing_image) - np.min(self.sensing_image)))
            self.bar1.setImageItem(self.spectrum_image_item, insert_in=self.spectrumPlotItem)  
            self.bar2.setImageItem(self.sensing_image_item, insert_in=self.sensingPlotItem)  

        # Set the axes of the channel occupancy map.     
        num_range = [*range(self.info['step_number']-self.render_history+1,self.info['step_number']+1)]
        x_range = list(map(str,num_range[::-1]))
        ticks = [(idx+0.5, label.rjust(6, ' ')) for idx, label in enumerate(x_range)]
        self.occupancyPlotItem.getAxis('left').setTicks((ticks[0::self.occupancy_tick_scale], []))
        if self.mode == 'iq':
            self.sensingPlotItem.getAxis('left').setTicks((ticks[0::self.sensing_tick_scale], []))
        num_range = [*range(0, self.num_channels)]
        y_range = list(map(str, num_range))        
        ticks = [(idx+0.5, label[0:4]) for idx, label in enumerate(y_range)]
        self.occupancyPlotItem.getAxis('bottom').setTicks((ticks, []))
        if self.mode == 'iq':
            self.sensingPlotItem.getAxis('bottom').setTicks((ticks, []))

        # Set the axes of the spectrum.
        if self.mode == 'iq':
            num_range = [*range(
                self.samples_per_step * (self.info['step_number']-self.render_history+1) + int(self.samples_per_step/2),
                self.samples_per_step*(self.info['step_number']+1)+int(self.samples_per_step/2),
                self.samples_per_step)]
            x_range = list(map(str,num_range[::-1]))
            ticks = [((idx+0.5)*self.spectrum_image.shape[1]/self.render_history, label) for idx, label in enumerate(x_range)]
            self.spectrumPlotItem.getAxis('left').setTicks((ticks[0::self.spectrum_tick_scale], []))
            y_range = list(map(str, np.round(1000.0*self.fc[0:self.num_channels])/1000.0))        
            ticks = [((float(label)+0.5)*self.spectrum_image.shape[0], label) for idx, label in enumerate(y_range)]
            self.spectrumPlotItem.getAxis('bottom').setTicks((ticks, []))

        # Update cumulative reward per step graph.
        for agent_id, color in zip(self.agents_info.keys(), self.entity_colors[self.num_entities:]): 
            color =  tuple(np.multiply(255, color))
            self.cummulative_reward_view.plot(
                self.info['cumulative_reward'][agent_id][0:self.info['step_number']],
                pen=color, penSize=1, symbol='o', symbolPen=color, symbolSize=2.5, symbolBrush=color)
     
        # Update cumulative reward per episode graph.        
        if self.num_episodes != 1:
            for agent_id, color  in zip(self.agents_info.keys(), self.entity_colors[self.num_entities:]): 
                self.episode_reward_view.plot(self.info['reward_history'][agent_id],
                    pen=color, penSize=1, symbol='o', symbolPen=color, symbolSize=2.5, symbolBrush=color)
                self.__add_padding_to_plot_widget(self.episode_reward_view, padding=0.01)
        
        if self.show_flag == 0:            
            self.show()
            self.__add_padding_to_plot_widget(self.cummulative_reward_view, padding=0.01)
            if self.num_episodes != 1:
                self.__add_padding_to_plot_widget(self.episode_reward_view, padding=0.01)
            self.show_flag = 1  
        
        QApplication.processEvents()

    def _reset(self):   
        self.occupancy_image = np.zeros([self.num_channels, self.render_history, 3], dtype=int)
        self.sensing_image = np.zeros([self.num_channels, self.render_history], dtype=float)
        self.cummulative_reward_view.clear()   

    def __initialize_occupancy_view(self):
        self.occupancy_tick_scale = self.__get_tick_scale(self.render_history)

        self.occupancy_view = pg.GraphicsLayoutWidget()

        self.occupancy_image_item = pg.ImageItem()
        self.grid_view = pg.GridItem()
        if self.render_background == "white":
            self.grid_view.setPen("black",width=4)
            self.occupancyPlotItem = self.occupancy_view.addPlot(row=0,col=0,lockAspect=False,
                                                                 border= pg.mkPen({'color': "#000000"}))            
        elif self.render_background == "black":  
            self.grid_view.setPen(pg.mkPen({'color': (255,255,255)}))  
            self.occupancyPlotItem = self.occupancy_view.addPlot(row=0,col=0,lockAspect=False,
                                                                 border= pg.mkPen({'color': "#FFFFFF"}))
        self.occupancyPlotItem.invertY(True)     
        self.occupancyPlotItem.setDefaultPadding(0.0) 
        self.occupancyPlotItem.setMouseEnabled(x=False,y=False)
        self.occupancyPlotItem.getAxis('bottom').setPen(self.pen_color)
        self.occupancyPlotItem.getAxis('bottom').setTextPen(self.pen_color)
        self.occupancyPlotItem.getAxis('left').setPen(self.pen_color)
        self.occupancyPlotItem.getAxis('left').setTextPen(self.pen_color)
        self.occupancyPlotItem.setLabels(bottom='Channel Number',left='Step Number')

        self.grid_view.setTextPen(None)         
        self.grid_view.setTickSpacing(x=[1.0],y=[1.0])  

        self.occupancyPlotItem.addItem(self.occupancy_image_item)
        self.occupancyPlotItem.addItem(self.grid_view) 
        self.occupancyPlotItem.showAxes(True)

    def __initialize_sensing_view(self):
        self.sensing_tick_scale = self.__get_tick_scale(self.render_history)

        self.sensing_view = pg.GraphicsLayoutWidget()
        self.grid_view1 = pg.GridItem()
        self.sensing_image_item = pg.ImageItem()
        if self.render_background == "white":
            self.sensingPlotItem = self.sensing_view.addPlot(row=0,
                col=0,lockAspect=False, border= pg.mkPen({'color': "#000000"}))            
        elif self.render_background == "black":  
            self.sensingPlotItem = self.sensing_view.addPlot(row=0,
                col=0,lockAspect=False, border= pg.mkPen({'color': "#FFFFFF"}))
        self.sensingPlotItem.invertY(True)     
        self.sensingPlotItem.setDefaultPadding(0.0) 
        self.sensingPlotItem.setMouseEnabled(x=False,y=False)
        self.sensingPlotItem.getAxis('bottom').setPen(self.pen_color)
        self.sensingPlotItem.getAxis('bottom').setTextPen(self.pen_color)
        self.sensingPlotItem.getAxis('left').setPen(self.pen_color)
        self.sensingPlotItem.getAxis('left').setTextPen(self.pen_color)
        self.sensingPlotItem.setLabels(bottom='Channel Number',left='Step Number')
        self.colormap1 = pg.colormap.getFromMatplotlib('nipy_spectral')
        self.bar1 = pg.ColorBarItem(cmap=self.colormap1, interactive=False)
        self.grid_view1.setPen(pg.mkPen({'color': (255,255,255)}))  
        self.grid_view1.setTextPen(None)         
        self.grid_view1.setTickSpacing(x=[1.0],y=[1.0])  

        self.sensing_image_item.setColorMap(self.colormap1)
        self.sensingPlotItem.addItem(self.sensing_image_item)
        self.sensingPlotItem.showAxes(True)
        self.sensingPlotItem.addItem(self.grid_view1) 

    def __initialize_spectrum_view(self):
        self.spectrum_tick_scale = self.__get_tick_scale(self.render_history)
        self.spectrum_view = pg.GraphicsLayoutWidget()
        self.spectrum_image_item = pg.ImageItem()
        if self.render_background == "white":
            self.spectrumPlotItem = self.spectrum_view.addPlot(row=0,
                col=0,lockAspect=False, border= pg.mkPen({'color': "#000000"}))            
        elif self.render_background == "black":  
            self.spectrumPlotItem = self.spectrum_view.addPlot(row=0,
                col=0,lockAspect=False, border= pg.mkPen({'color': "#FFFFFF"}))
        self.spectrumPlotItem.invertY(True)
        self.spectrumPlotItem.setDefaultPadding(0.0) 
        self.spectrumPlotItem.setMouseEnabled(x=False,y=False)
        self.spectrumPlotItem.getAxis('bottom').setPen(self.pen_color)
        self.spectrumPlotItem.getAxis('bottom').setTextPen(self.pen_color)
        self.spectrumPlotItem.getAxis('left').setPen(self.pen_color)
        self.spectrumPlotItem.getAxis('left').setTextPen(self.pen_color)
        self.spectrumPlotItem.addItem(self.spectrum_image_item)
        self.spectrumPlotItem.showAxes(True)
        self.colormap2 = pg.colormap.getFromMatplotlib('nipy_spectral')
        self.bar2 = pg.ColorBarItem(cmap=self.colormap2, interactive=False)

        self.spectrum_image_item.setColorMap(self.colormap2)
        self.spectrumPlotItem.setLabels(bottom='Frequency (Hz)',left='Time (samples)')

    def __initialize_logo_view(self):
        self.logo_view = pg.GraphicsLayoutWidget()
        self.logo_image_item = pg.ImageItem()
        if self.render_background == "white":
            self.logoPlotItem = self.logo_view.addPlot(row=0,
                col=0,lockAspect=True, border= pg.mkPen({'color': "#000000"}))            
        elif self.render_background == "black":  
            self.logoPlotItem = self.logo_view.addPlot(row=0,
                col=0,lockAspect=True, border= pg.mkPen({'color': "#FFFFFF"}))
        im = plt.imread('logowhite.png', format='png')
        self.logo_image_item.setImage(np.rot90(im,k=3))
        self.logo_view.setFixedSize(im.shape[1],im.shape[0])
        self.logo_view.ci.setContentsMargins(0, 0, 0, 0)
        self.logoPlotItem.addItem(self.logo_image_item)
        self.logoPlotItem.setMouseEnabled(x=False,y=False)
        self.logoPlotItem.hideAxis('left')
        self.logoPlotItem.hideAxis('bottom')
        self.logoPlotItem.setDefaultPadding(0.0) 

    def __initialize_legend_view(self):
        self.legend_view = pg.TreeWidget()
        self.legend_view.setDragEnabled(False)
        self.legend_view.setStyleSheet("background-color: " + str(self.render_background) + "; color: black; border: none;")
        self.legend_view.setRootIsDecorated(False)
        self.legend_view.setUniformRowHeights(True)
        self.legend_view.setAllColumnsShowFocus(True)
        self.legend_view.setItemsExpandable(False)
        self.legend_view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.legend_view.header().hide()
        self.legend_view.header().setStretchLastSection(False)
        self.legend_view.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.legend_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.legend_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.legend_view.setColumnCount(1)
        self.entity_colors = distinctipy.get_colors(self.num_entities + self.num_agents,
            [(0,0,0),(1,1,1),(1,0,0),(0,1,0),(0,0,1),(1,1,0)], rng=self.random_seed)

        # In the self.entity_colors list, the first portion corresponds to entities
        # and the agents' colors come after.
        legend = []

        # Adds the agents to the legend
        for agent_num, (agent_id, info) in enumerate(self.agents_info.items()):
            agent_color = self.entity_colors[-1*self.num_agents+agent_num]
            agent_color = tuple([int(255*a_color) for a_color in agent_color])
            legend.append((f'{agent_id} (Agent)', agent_color)) # Gives a color to the specific agent

        # Adds the other things to the legend
        if self.observation_mode == 'classify':
            entity_idx = 0
            for entity in self.entity_list:
                entity_idx += 1
                label = str(entity) + ' (Entity)'
                color = np.multiply(255, self.entity_colors[entity_idx-1])                
                legend.append((label,color))
            legend.append(('Collision Between Entities','yellow'))
            legend.append((f'Collision Involving Agent(s)','red'))
        elif self.observation_mode == 'detect':
            legend.append(('Non-Empty Channel','blue'))

        item = QTreeWidgetItem(self.legend_view)
        item.setDisabled(True)
        for entity_label, entity_color in (legend):
            item = QTreeWidgetItem(self.legend_view)
            pixmap = QtGui.QPixmap(96, 96)
            painter = QtGui.QPainter(pixmap)
            if isinstance(entity_color,str):
                painter.setBrush(QtGui.QColor(entity_color))
                if self.render_background == 'black':
                    item.setForeground(0, QtGui.QBrush(QtGui.QColor('white')))
                else:
                    item.setForeground(0, QtGui.QBrush(QtGui.QColor('black')))
            else:
                painter.setBrush(QtGui.QColor(int(entity_color[0]),int(entity_color[1]),int(entity_color[2])))
                if self.render_background == 'black':
                    item.setForeground(0, QtGui.QBrush(QtGui.QColor(255,255,255)))
                else:
                    item.setForeground(0, QtGui.QBrush(QtGui.QColor(0,0,0)))
            painter.drawRect(pixmap.rect())
            painter.end()
            item.setIcon(0,QtGui.QIcon(pixmap))
            item.setText(0,entity_label)

    def __initialize_cummulative_reward_view(self):
        self.cummulative_reward_view = pg.PlotWidget(lockAspect=False)
        self.cummulative_reward_view.setMouseEnabled(x=False,y=False)

         # Set up the cumulative reward per step widget.
        tick_scale = self.__get_tick_scale(self.max_steps)
        self.cummulative_reward_view.setTitle("Cumulative Reward per Step", color=self.pen_color)
        self.cummulative_reward_view.setXRange(1,self.max_steps)
        self.cummulative_reward_view.setYRange(-self.max_steps,self.max_steps)
        self.cummulative_reward_view.setLabels(bottom='Step Number',left='Cumulative Reward')
        ticks_x = [*range(0,self.max_steps+1,tick_scale)]
        self.cummulative_reward_view.getAxis('bottom').setTicks([[(x,str(x)) for x in ticks_x]])
        self.cummulative_reward_view.getAxis('bottom').setPen(self.pen_color)
        self.cummulative_reward_view.getAxis('bottom').setTextPen(self.pen_color)
        ticks_y = [*range(-self.max_steps,self.max_steps+20,20)]
        self.cummulative_reward_view.getAxis('left').setTicks([[(x,str(x)) for x in ticks_y]])
        self.cummulative_reward_view.getAxis('left').setPen(self.pen_color)
        self.cummulative_reward_view.getAxis('left').setTextPen(self.pen_color)
        self.cummulative_reward_view.showAxes(True)

    def __initialize_episode_reward_view(self):
        self.episode_reward_view = pg.PlotWidget(lockAspect=False)
        self.episode_reward_view.setMouseEnabled(x=False,y=False)
        tick_scale = self.__get_tick_scale(self.num_episodes)
        self.episode_reward_view.setTitle("Cumulative Reward per Episode", color=self.pen_color)
        self.episode_reward_view.setXRange(0,self.num_episodes-1)
        self.episode_reward_view.setYRange(-self.max_steps,self.max_steps)
        self.episode_reward_view.setLabels(bottom='Episode Number',left='Cumulative Reward')
        ticks_x = [*range(0,self.num_episodes+1,tick_scale)]
        self.episode_reward_view.getAxis('bottom').setTicks([[(x,str(x)) for x in ticks_x]])
        self.episode_reward_view.getAxis('bottom').setPen(self.pen_color)
        self.episode_reward_view.getAxis('bottom').setTextPen(self.pen_color)
        ticks_y = [*range(-self.max_steps,self.max_steps+20,20)]
        self.episode_reward_view.getAxis('left').setTicks([[(x,str(x)) for x in ticks_y]])
        self.episode_reward_view.getAxis('left').setPen(self.pen_color)
        self.episode_reward_view.getAxis('left').setTextPen(self.pen_color) 
        self.episode_reward_view.showAxes(True)
        
    def __add_padding_to_plot_widget(self, plot_widget, padding=0.05):
        width = plot_widget.sceneRect().width()
        height = plot_widget.sceneRect().height()
        center = plot_widget.sceneRect().center()
        zoom_rect = QtCore.QRectF(center.x(), center.y(), width*(1.0+padding), height)
        plot_widget.fitInView(zoom_rect)

    def __get_occupancy_image(self):
        self.occupancy_image = np.roll(self.occupancy_image, 1, axis=1)
        for channel in range(self.num_channels):
            self.occupancy_image[channel][0] = np.zeros(3)
            if self.render_background == "white":
                self.occupancy_image[channel,0,:] = [255,255,255]
            elif self.render_background == "black":
                self.occupancy_image[channel,0,:] = [0,0,0]
            channel_entity = self.info['true_history'][self.info['step_number']][channel]

            # Checks if the channel is non-empty.
            if channel_entity > 0:
                if self.observation_mode == "detect":
                    self.occupancy_image[channel, 0, :] = [0, 0, 255] # Blue

                elif self.observation_mode == "classify":
                    # Here, there is no collision.
                    if channel_entity <= self.num_entities + self.num_agents:
                        self.occupancy_image[channel, 0, :] = np.multiply(255, self.entity_colors[channel_entity-1])

                    # Here, there is a collision.
                    else:
                        involves_an_agent = False
                        for agent_id in self.agents_info:
                            if self.info["action_history"][agent_id][0][self.info["step_number"]] == channel:
                                involves_an_agent = True
                                break
                        if involves_an_agent:
                            self.occupancy_image[channel, 0, :] = [255, 0, 0] # Red
                        else:
                            self.occupancy_image[channel, 0, :] = [255, 255, 0] # Yellow

    def __get_spectrogram_image(self):
        _, _, image = signal.spectrogram(self.info['spectrum_data'], nperseg=512, noverlap=0, return_onesided=False)
        self.spectrum_image = fftshift(10.0*np.log10(image), axes=0)

    def __get_sensing_image(self):
        self.sensing_image = np.roll(self.sensing_image, 1, axis=1)
        for k in range(self.num_channels):
            data = self.info['spectrum_data'][0:self.samples_per_step] * np.exp(-1j*2*np.pi*self.fc[k]*self.t)      
            filtered = signal.sosfilt(self.sos, data)
            downsampled = filtered[::self.num_channels]
            self.sensing_image[k, 0] = np.sum(np.abs(downsampled)**2.0)/(self.samples_per_step/self.num_channels)

    def __get_tick_scale(self, num_indices):
        done = 0
        tick_scale = 1
        while done == 0:                
            value = int(num_indices/(10*tick_scale))
            if value >= 10:
                tick_scale = 10*tick_scale
            else:
                done = 1
        return tick_scale
