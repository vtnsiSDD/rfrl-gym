import distinctipy
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QTreeWidget, QTreeWidgetItem, QHeaderView
from rfrl_gym.renderers.renderer import Renderer

class PyQtRenderer(Renderer, QMainWindow):
    def __init__(self, num_episodes, scenario_metadata):
        super(PyQtRenderer, self).__init__(num_episodes, scenario_metadata)
        self.max_steps = scenario_metadata['environment']['max_steps']
        self.render_background = scenario_metadata['render']['render_background']
        self.show_flag = 0

        # Initialize the main window.
        self.win_width = 1000
        self.win_height = 1000
        if self.render_background == 'white':
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')
            background = "background-color: " + str(self.render_background) + "; color: white;"
        elif self.render_background == 'black':
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')
            background = "background-color: " + str(self.render_background) + "; color: black;" 

        self.setWindowTitle('RFRL GYM')
        self.setFixedWidth(self.win_width)
        self.setFixedHeight(self.win_height)
        self.main_panel = QGridLayout()
        self.main_panel.setRowStretch(0,2)
        self.main_panel.setRowStretch(2,1)
        if self.num_episodes != 1:
            self.main_panel.setRowStretch(3,1)
        self.main_panel.setColumnStretch(0,4)
        self.main_panel.setColumnStretch(3,1)
        
        # Initialize the panels.
        self.spectrum_view = pg.GraphicsLayoutWidget()
        self.legend_view = QTreeWidget()
        self.legend_view.setStyleSheet(background)
        self.legend_view.setRootIsDecorated(False)
        self.legend_view.setUniformRowHeights(True)
        self.legend_view.setAllColumnsShowFocus(True)
        self.legend_view.setItemsExpandable(False)
        self.legend_view.header().hide()
        self.legend_view.header().setStretchLastSection(False)
        self.legend_view.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.legend_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.legend_view.setColumnCount(1)
        self.cummulative_reward_view = pg.PlotWidget(lockAspect=False)
        self.cummulative_reward_view.setMouseEnabled(x=False,y=False)
        
        self.main_panel.addWidget(self.spectrum_view,                                0, 0, 1, 3)
        self.main_panel.addWidget(self.legend_view,                                  0, 3, 1, 1)
        self.main_panel.addWidget(self.cummulative_reward_view,                      2, 0, 1, 4)

        if self.num_episodes != 1:
            self.episode_reward_view = pg.PlotWidget(lockAspect=False)
            self.episode_reward_view.setMouseEnabled(x=False,y=False)
            self.main_panel.addWidget(self.episode_reward_view,                      3, 0, 1, 4)
        
        # Finalize the main window.
        self.widget = QWidget()
        self.widget.setLayout(self.main_panel)
        self.setCentralWidget(self.widget)
        self.add_widgets()
        self.add_legend()

    def add_widgets(self):
        # Initialize the spectrum widget.
        self.spectrum_image = pg.ImageItem()
        self.grid_view = pg.GridItem()
        if self.render_background == "white":
            pen_color = 'k'
            self.grid_view.setPen("black",width=2)
            self.plotItem = self.spectrum_view.addPlot(row=0,col=0,lockAspect=False, border= pg.mkPen({'color': "#000000"}))
        elif self.render_background == "black":  
            pen_color = 'w'
            self.grid_view.setPen("#FFFFFF",width=2,cosmetic=False)  
            self.plotItem = self.spectrum_view.addPlot(row=0,col=0,lockAspect=False, border= pg.mkPen({'color': "#FFFFFF"}))
        self.grid_view.setTextPen(None)         
        self.grid_view.setTickSpacing(x=[1.0],y=[1.0])  
        self.plotItem.invertY(True)     
        self.plotItem.setDefaultPadding(0.0) 
        self.plotItem.setMouseEnabled(x=False,y=False)
        self.plotItem.getAxis('bottom').setPen(pen_color)
        self.plotItem.getAxis('bottom').setTextPen(pen_color)
        self.plotItem.getAxis('left').setPen(pen_color)
        self.plotItem.getAxis('left').setTextPen(pen_color)
        self.plotItem.addItem(self.spectrum_image)
        self.plotItem.addItem(self.grid_view) 
        self.plotItem.showAxes(True)

        # Set up the cumulative reward per step widget.
        tick_scale = self.__get_tick_scale(self.max_steps)
        self.cummulative_reward_view.setTitle("Cumulative Reward per Step", color=pen_color)
        self.cummulative_reward_view.setXRange(1,self.max_steps)
        self.cummulative_reward_view.setYRange(-self.max_steps,self.max_steps)
        self.cummulative_reward_view.setLabels(bottom='Step Number',left='Cumulative Reward')
        ticks_x = [*range(0,self.max_steps+1,tick_scale)]
        self.cummulative_reward_view.getAxis('bottom').setTicks([[(x,str(x)) for x in ticks_x]])
        self.cummulative_reward_view.getAxis('bottom').setPen(pen_color)
        self.cummulative_reward_view.getAxis('bottom').setTextPen(pen_color)
        ticks_y = [*range(-self.max_steps,self.max_steps+20,20)]
        self.cummulative_reward_view.getAxis('left').setTicks([[(x,str(x)) for x in ticks_y]])
        self.cummulative_reward_view.getAxis('left').setPen(pen_color)
        self.cummulative_reward_view.getAxis('left').setTextPen(pen_color)
        self.cummulative_reward_view.showAxes(True)

        # Set up the cummulative reward per episode widget.
        if self.num_episodes != 1:
            tick_scale = self.__get_tick_scale(self.num_episodes)
            self.episode_reward_view.setTitle("Cumulative Reward per Episode", color=pen_color)
            self.episode_reward_view.setXRange(0,self.num_episodes-1)
            self.episode_reward_view.setYRange(-self.max_steps,self.max_steps)
            self.episode_reward_view.setLabels(bottom='Episode Number',left='Cumulative Reward')
            ticks_x = [*range(0,self.num_episodes+1,tick_scale)]
            self.episode_reward_view.getAxis('bottom').setTicks([[(x,str(x)) for x in ticks_x]])
            self.episode_reward_view.getAxis('bottom').setPen(pen_color)
            self.episode_reward_view.getAxis('bottom').setTextPen(pen_color)
            ticks_y = [*range(-self.max_steps,self.max_steps+20,20)]
            self.episode_reward_view.getAxis('left').setTicks([[(x,str(x)) for x in ticks_y]])
            self.episode_reward_view.getAxis('left').setPen(pen_color)
            self.episode_reward_view.getAxis('left').setTextPen(pen_color) 
            self.episode_reward_view.showAxes(True)

    def add_legend(self):
        self.entity_colors = distinctipy.get_colors(self.num_entities, [(0,0,0),(1,1,1),(1,0,0),(0,1,0),(0,0,1),(1,1,0)])

        legend = []
        legend.append(('Player',(0,255,0)))
        if self.observation_mode == 'classify':
            entity_idx = 0
            for entity in self.entity_list:
                entity_idx += 1
                label = str(entity) + ' Entity'
                color = np.multiply(255, self.entity_colors[entity_idx-1])                
                legend.append((label,color))
            legend.append(('Multi-Entity Collision','yellow'))
            legend.append(('Player Collision','red'))
        elif self.observation_mode == 'detect':
            legend.append(('Entities','blue'))
            legend.append(('Player Collision w/Entity','red'))

        for entity_label, entity_color in (legend):
            item = QTreeWidgetItem()
            pixmap = QtGui.QPixmap(96, 96)
            painter = QtGui.QPainter(pixmap)
            if isinstance(entity_color,str):
                painter.setBrush(QtGui.QColor(entity_color))
                item.setForeground(0, QtGui.QBrush(QtGui.QColor('white')))
            else:
                painter.setBrush(QtGui.QColor(int(entity_color[0]),int(entity_color[1]),int(entity_color[2])))
                item.setForeground(0, QtGui.QBrush(QtGui.QColor(255,255,255)))
            painter.drawRect(pixmap.rect())
            painter.end()
            item.setIcon(0,QtGui.QIcon(pixmap))
            item.setText(0,entity_label)
            self.legend_view.addTopLevelItem(item)

    def _render(self):
        # Clear and update the channel occupancy map.  
        self.spectrum_image.clear() 
        self.__get_occupancy_image()     
        self.spectrum_image.setImage(self.occupancy_image)  
        
        # Set the axes of the channel occupancy map.     
        tick_scale = self.__get_tick_scale(self.render_history)
        self.plotItem.setLabels(bottom='Channel Number',left='Step Number')
        self.plotItem.setTitle('Occupancy Map')
        num_range = [*range(self.info['step_number']-self.render_history+1,self.info['step_number']+1)]
        x_range = list(map(str,num_range[::-1]))
        ticks = [(idx+0.5, label) for idx, label in enumerate(x_range)]
        self.plotItem.getAxis('left').setTicks((ticks[0::tick_scale], []))
        num_range = [*range(0, self.num_channels)]
        y_range = list(map(str, num_range))        
        ticks = [(idx+0.5, label) for idx, label in enumerate(y_range)]
        self.plotItem.getAxis('bottom').setTicks((ticks, []))

        # Update cumulative reward per step graph.
        self.cummulative_reward_view.plot(self.info['cumulative_reward'][0:self.info['step_number']],pen=(0,255,0), symbol='o', symbolPen=(0,255,0), symbolSize=2.5, symbolBrush=(0,255,0))
        self.cummulative_reward_view.showAxes(True)

        # Update cumulative reward per episode graph.        
        if self.info['step_number'] == self.max_steps and self.num_episodes > 1:
            self.episode_reward_view.plot(self.info['episode_reward'],pen=(0,255,0), symbol='o', symbolPen=(0,255,0), symbolSize=2.5, symbolBrush=(0,255,0))
            self.episode_reward_view.showAxes(True)
        
        if self.show_flag == 0:            
            self.show()
            self.__add_padding_to_plot_widget(self.spectrum_view)
            self.__add_padding_to_plot_widget(self.cummulative_reward_view)
            if self.num_episodes > 1:
                self.__add_padding_to_plot_widget(self.episode_reward_view) 
            self.show_flag = 1
                
        QApplication.processEvents()

    def _reset(self):
        self.occupancy_image = np.zeros([self.num_channels, self.render_history, 3], dtype=int)
        self.cummulative_reward_view.clear()   

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
            channel_entity = self.info['observation_history'][self.info['step_number']][channel]
            if self.info['action_history'][self.info['step_number']] == channel:
                if channel_entity != 0:
                    self.occupancy_image[channel, 0, :] = [255, 0, 0]
                else:
                    self.occupancy_image[channel, 0, :] = [0, 255, 0]
            else:
                if self.observation_mode == 'detect' and channel_entity != 0:
                    self.occupancy_image[channel, 0, :] = [0, 0, 255]
                elif self.observation_mode == 'classify' and channel_entity != 0:
                    if channel_entity == self.num_entities + 1:
                        self.occupancy_image[channel, 0, :] = [255, 255, 0]
                    else:
                        self.occupancy_image[channel, 0, :] = np.multiply(255, self.entity_colors[channel_entity-1])

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