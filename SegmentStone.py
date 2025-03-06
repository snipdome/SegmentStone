# 
# This file is part of the SegmentStone distribution (https://github.com/snipdome/SegmentStone).
# Copyright (c) 2024 imec-Vision Lab, University of Antwerp.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
 
import logging
import os
from typing import Annotated, Optional

import vtk
import vtkSegmentationCorePython as vtkSegmentationCore 
import vtkSlicerSegmentationsModuleLogicPython as vtkSlicerSegmentationsModuleLogic
import vtkITK

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
	parameterNodeWrapper,
	WithinRange,
)
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode

import numpy as np
import time
import threading
from enum import Enum

from skimage.segmentation import flood, flood_fill #FIXME remove these imports
from matplotlib import pyplot as plt #FIXME remove these imports

import SimpleITK as sitk
from vtk.util import numpy_support
import sitkUtils

#
# SegmentStone
#


class SegmentStone(ScriptedLoadableModule):

	def __init__(self, parent):
		ScriptedLoadableModule.__init__(self, parent)
		self.parent.title = _("SegmentStone")  
		self.parent.categories = [translate("qSlicerAbstractCoreModule", "Post-Process")]
		self.parent.dependencies = []
		self.parent.contributors = ["Domenico Iuso (imec-Visionlab, UAntwerp)"] 
		self.parent.helpText = _("""
This is a scripted loadable module for the segmentation of glued stones in X-ray CT images.
See more information in <a href="https://github.com/snipdome/SegmentStone">module documentation</a>.
""")
		self.parent.acknowledgementText = _("""
This file was originally developed by Domenico Iuso (imec-Visionlab, UAntwerp). 
""")

		# Additional initialization step after application startup is complete
		slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
	"""Add data sets to Sample Data module."""
	# It is always recommended to provide sample data for users to make it easy to try the module,
	# but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

	import SampleData

	iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

	# To ensure that the source code repository remains small (can be downloaded and installed quickly)
	# it is recommended to store data sets that are larger than a few MB in a Github release.

	# SegmentStone1
	SampleData.SampleDataLogic.registerCustomSampleDataSource(
		# Category and sample name displayed in Sample Data module
		category="SegmentStone",
		sampleName="SegmentStone1",
		# Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
		# It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
		thumbnailFileName=os.path.join(iconsPath, "SegmentStone1.png"),
		# Download URL and target file name
		uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
		fileNames="SegmentStone1.nrrd",
		# Checksum to ensure file integrity. Can be computed by this command:
		#  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
		checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
		# This node name will be used when the data set is loaded
		nodeNames="SegmentStone1",
	)

	# SegmentStone2
	SampleData.SampleDataLogic.registerCustomSampleDataSource(
		# Category and sample name displayed in Sample Data module
		category="SegmentStone",
		sampleName="SegmentStone2",
		thumbnailFileName=os.path.join(iconsPath, "SegmentStone2.png"),
		# Download URL and target file name
		uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
		fileNames="SegmentStone2.nrrd",
		checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
		# This node name will be used when the data set is loaded
		nodeNames="SegmentStone2",
	)


#
# SegmentStoneParameterNode
#


@parameterNodeWrapper
class SegmentStoneParameterNode:
	"""
	The parameters needed by module.

	inputVolume - The volume to threshold.
	outputLabelMap - The output label map, where the segmented volume will be written.
	"""

	inputVolume: vtkMRMLScalarVolumeNode
	outputLabelMap: vtkMRMLSegmentationNode
	#imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
	


#
# SegmentStoneWidget
#


class SegmentStoneWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
	"""Uses ScriptedLoadableModuleWidget base class, available at:
	https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def __init__(self, parent=None) -> None:
		"""Called when the user opens the module the first time and the widget is initialized."""
		ScriptedLoadableModuleWidget.__init__(self, parent)
		VTKObservationMixin.__init__(self)  # needed for parameter node observation
		self.logic = None
		self._parameterNode = None
		self._parameterNodeGuiTag = None

	def setup(self) -> None:
		"""Called when the user opens the module the first time and the widget is initialized."""
		ScriptedLoadableModuleWidget.setup(self)

		# Load widget from .ui file (created by Qt Designer).
		# Additional widgets can be instantiated manually and added to self.layout.
		uiWidget = slicer.util.loadUI(self.resourcePath("UI/SegmentStone.ui"))
		self.layout.addWidget(uiWidget)
		self.ui = slicer.util.childWidgetVariables(uiWidget)

		# hide the debug level slider
		self.ui.debugLevelSliderWidget.hide()
		self.ui.debugLevelLabel.hide()

		# hide the curvature flow options
		self.ui.curvatureFlowGroupBox.hide()

		# Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
		# "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
		# "setMRMLScene(vtkMRMLScene*)" slot.
		uiWidget.setMRMLScene(slicer.mrmlScene)

		# Create logic class. Logic implements all computations that should be possible to run
		# in batch mode, without a graphical user interface.
		self.logic = SegmentStoneLogic()

		# Connections

		# These connections ensure that we update parameter node when scene is closed
		self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
		self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

		# Buttons
		self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

		# Make sure parameter node is initialized (needed for module reload)
		self.initializeParameterNode()

	def cleanup(self) -> None:
		"""Called when the application closes and the module widget is destroyed."""
		self.removeObservers()

	def enter(self) -> None:
		"""Called each time the user opens this module."""
		# Make sure parameter node exists and observed
		self.initializeParameterNode()

	def exit(self) -> None:
		"""Called each time the user opens a different module."""
		# Do not react to parameter node changes (GUI will be updated when the user enters into the module)
		if self._parameterNode:
			self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
			self._parameterNodeGuiTag = None
			self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

	def onSceneStartClose(self, caller, event) -> None:
		"""Called just before the scene is closed."""
		# Parameter node will be reset, do not use it anymore
		self.setParameterNode(None)

	def onSceneEndClose(self, caller, event) -> None:
		"""Called just after the scene is closed."""
		# If this module is shown while the scene is closed then recreate a new parameter node immediately
		if self.parent.isEntered:
			self.initializeParameterNode()

	def initializeParameterNode(self) -> None:
		"""Ensure parameter node exists and observed."""
		# Parameter node stores all user choices in parameter values, node selections, etc.
		# so that when the scene is saved and reloaded, these settings are restored.

		self.setParameterNode(self.logic.getParameterNode())

		# Select default input nodes if nothing is selected yet to save a few clicks for the user
		if not self._parameterNode.inputVolume:
			firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
			if firstVolumeNode:
				self._parameterNode.inputVolume = firstVolumeNode
		
		self.try_update_output_labelmap()
		
	def try_update_output_labelmap(self):
		# Create a new segmentation node to store the output if there is none yet with the same name as the input volume
		if not self._parameterNode.outputLabelMap and self._parameterNode.inputVolume:
			input_name = self._parameterNode.inputVolume.GetName().split("_")[0] 
			segmentationNode = slicer.util.getFirstNodeByClassByName("vtkMRMLSegmentationNode", input_name)
			if not segmentationNode:
				segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
				segmentationNode.SetName(input_name)
			self._parameterNode.outputLabelMap = segmentationNode


	def setParameterNode(self, inputParameterNode: Optional[SegmentStoneParameterNode]) -> None:
		"""
		Set and observe parameter node.
		Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
		"""

		if self._parameterNode:
			self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
			self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
		self._parameterNode = inputParameterNode
		if self._parameterNode:
			# Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
			# ui element that needs connection.
			self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
			self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
			self._checkCanApply()

	def _checkCanApply(self, caller=None, event=None) -> None:
		self.try_update_output_labelmap()
		if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.outputLabelMap:
			self.ui.applyButton.toolTip = _("Compute segmentation")
			self.ui.applyButton.enabled = True
		else:
			self.ui.applyButton.toolTip = _("Select input volume and output labelmap nodes")
			self.ui.applyButton.enabled = False

	def onApplyButton(self) -> None:

		if self.ui.borderDilateSliderWidget.value <= self.ui.scaleSliderWidget.value:
			self.ui.borderDilateSliderWidget.value = self.ui.scaleSliderWidget.value +1
			logging.warning("Border dilate radius should be greater than the scale. Setting it to scale + 1")
		"""Run processing when user clicks "Apply" button."""
		with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
			kwargs = {
				"inputVolume": self.ui.inputSelector.currentNode(),
				"outputLabelMap": self.ui.outputSelector.currentNode(),

				"scale": self.ui.scaleSliderWidget.value,
				"want_curvature_flow": self.ui.curvatureFlowCheckBox.checked,
				"curvature_niter": self.ui.iterationsCVSliderWidget.value,
				"curvature_timestep": self.ui.stepSizeCVSliderWidget.value,

				"top_stone_seed_perc": self.ui.topStoneHeightSliderWidget.value,
				"bottom_stone_seed_perc": self.ui.bottomStoneHeightSliderWidget.value,
				"border_dilate_radius": self.ui.borderDilateSliderWidget.value,
				"border_erode_radius": self.ui.borderErosionSliderWidget.value,
				"seed_radius": self.ui.seedRadiusSliderWidget.value,
				"glue_h": self.ui.glueHeightSliderWidget.value,
				"debug_level": self.ui.debugLevelSliderWidget.value,
			}

			# Compute output
			self.logic.process(**kwargs)

#
# SegmentStoneLogic
#


class SegmentStoneLogic(ScriptedLoadableModuleLogic):
	"""This class should implement all the actual
	computation done by your module.  The interface
	should be such that other python code can import
	this class and make use of the functionality without
	requiring an instance of the Widget.
	Uses ScriptedLoadableModuleLogic base class, available at:
	https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def __init__(self) -> None:
		"""Called when the logic class is instantiated. Can be used for initializing member variables."""
		ScriptedLoadableModuleLogic.__init__(self)

	def getParameterNode(self):
		return SegmentStoneParameterNode(super().getParameterNode())

	def process(self,
				inputVolume: vtkMRMLScalarVolumeNode,
				outputLabelMap: vtkMRMLSegmentationNode,
				scale: int = 2,
				want_curvature_flow: bool = True,
				curvature_niter: int = 5,
				curvature_timestep: float = 1.e-5,
				bottom_stone_seed_perc: float = 0.1,
				top_stone_seed_perc: float = 0.9,
				border_dilate_radius: int = 4,
				border_erode_radius: int = 3,
				seed_radius: int = 7,
				glue_h: int = 20,
				debug_level: int = 0,
				) -> None:
		"""
		Run the processing algorithm.
		Can be used without GUI widget.
		:param inputVolume: volume to be thresholded
		:param outputLabelMap: output label map where the segmented volume will be written
		:param modify_input: if True then input volume will be modified, if False nothing will be changed
		
		"""

		if not inputVolume or not outputLabelMap:
			raise ValueError("Input or output volume is invalid")

		import time

		startTime = time.time()
		logging.info("Processing started")

		# Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
		cliParams = {
			"inputVolume": inputVolume.GetID(),
			"outputLabelMap": outputLabelMap.GetID(),
			"scale": int(scale),
			"want_curvature_flow": want_curvature_flow,
			"curvature_niter": int(curvature_niter),
			"curvature_timestep": curvature_timestep,
			"bottom_stone_seed_perc": bottom_stone_seed_perc,
			"top_stone_seed_perc": top_stone_seed_perc,
			"border_dilate_radius": int(border_dilate_radius),
			"border_erode_radius": int(border_erode_radius),
			"seed_radius": int(seed_radius),
			"glue_h": int(glue_h),
			"debug_level": int(debug_level),
		}
		#print(f"cliParams: {cliParams}")

		# Dome
		# cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
		# # We don't need the CLI module node anymore, remove it to not clutter the scene with it
		# slicer.mrmlScene.RemoveNode(cliNode)
		segmentation_kernel(**cliParams)


		stopTime = time.time()
		logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# SegmentStoneTest
#


class SegmentStoneTest(ScriptedLoadableModuleTest):
	"""
	This is the test case for your scripted module.
	Uses ScriptedLoadableModuleTest base class, available at:
	https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def setUp(self):
		"""Do whatever is needed to reset the state - typically a scene clear will be enough."""
		slicer.mrmlScene.Clear()

	def runTest(self):
		"""Run as few or as many tests as needed here."""
		self.setUp()
		self.test_SegmentStone1()

	def test_SegmentStone1(self):
		"""Ideally you should have several levels of tests.  At the lowest level
		tests should exercise the functionality of the logic with different inputs
		(both valid and invalid).  At higher levels your tests should emulate the
		way the user would interact with your code and confirm that it still works
		the way you intended.
		One of the most important features of the tests is that it should alert other
		developers when their changes will have an impact on the behavior of your
		module.  For example, if a developer removes a feature that you depend on,
		your test should break so they know that the feature is needed.
		"""
		self.delayDisplay("Starting the test")
		import os
		if os.name == 'nt':
			dataset_path = ''
			raise ValueError("Fill in the dataset path")
		else:
			dataset_path = ''
			raise ValueError("Fill in the dataset path")
		masterVolumeNode = slicer.util.loadVolume(dataset_path + '.tif')
		outputLabelMap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

		self.delayDisplay("Loaded test data set")

		kwargs = {
			"inputVolume": masterVolumeNode,
			"outputLabelMap": outputLabelMap,
			"scale": 4,
			"want_curvature_flow": False,
			"debug_level": 0,
		}
		# Test the module logic
		logic = SegmentStoneLogic()
		logic.process(**kwargs)
		

## debug: 0 no verbose // 1 just text verbose // 2 show intermediate segmentations // 3: save images
class DebugLevel(Enum):
	NO_VERBOSE = 0
	TEXT_VERBOSE = 1
	SHOW_INTERM_SEGMS = 2
	SAVE_IMAGES = 3

colours = {
	"Background": [0.0,0.0,1.0],
	"Glue": [1.0,0.6,0.],
	"Stone": [1.0,1.0,0.3],
	"Border": [1.0,0.0,0.]
}


def deleteNode(node_to_delete):
	if slicer.mrmlScene:
		tmpdict = slicer.util.getNodes(useLists=True)
		for name,tmplist in tmpdict.items():
			for node in tmplist:
				if node.GetName() == node_to_delete.GetName():
					slicer.mrmlScene.RemoveNode(node)

def detect_bounding_box(proj, seedpoint,savepath=None): # proj is a 2D numpy array
	if savepath is not None:
		plt.imshow(proj)
		plt.savefig(savepath)
	plt.close()
	proj2 = flood(proj, seedpoint, connectivity=1)
	if savepath is not None:
		plt.imshow(proj2)
		plt.savefig(savepath)
	x_min = np.where(proj2.sum(axis=0))[0][0]
	x_max = np.where(proj2.sum(axis=0))[0][-1]
	y_min = np.where(proj2.sum(axis=1))[0][0]
	y_max = np.where(proj2.sum(axis=1))[0][-1]
	if savepath is not None:
		plt.imshow(proj2)
		# draw the bounding box
		plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'r')
		plt.savefig(savepath.replace(".png", "_2.png"))
		plt.close()
	return x_min, x_max, y_min, y_max # in matplotlib, the first two coordinates goes along the horizontal axis, the second two along the vertical axis

def find_minimal_bounding_box(volume, z_stone_center, stone_value, debug_level=DebugLevel.NO_VERBOSE.value):
	# get the bounding box of the stone 
	# by projecting the maximum along the three axes
	# volume between 0.8 and 1.2  of the stone value is considered stone
	bool_volume = (volume > 0.8*stone_value) & (volume < 1.2*stone_value)
	x_proj = np.max(bool_volume, axis=0).astype(int)
	y_proj = np.max(bool_volume, axis=1).astype(int)
	z_proj = np.max(bool_volume, axis=2).astype(int)

	seedpoint_x = (z_stone_center[1], z_stone_center[2])
	z_minx, z_maxx, y_minx, y_maxx = detect_bounding_box(x_proj, seedpoint_x, "proj_x.png" if debug_level>=DebugLevel.SAVE_IMAGES.value else None)
	# print(f'Seedpoint: {seedpoint_x}')
	# print(f'stone bounding box (x): {y_minx}:{y_maxx}, {z_minx}:{z_maxx}')
	seedpoint_y = (z_stone_center[0], z_stone_center[2])
	z_miny, z_maxy, x_miny, x_maxy = detect_bounding_box(y_proj, seedpoint_y, "proj_y.png" if debug_level>=DebugLevel.SAVE_IMAGES.value else None)
	# print(f'Seedpoint: {seedpoint_y}')
	# print(f'stone bounding box (y): {x_miny}:{x_maxy}, {z_miny}:{z_maxy}')
	seedpoint_z = (z_stone_center[0], z_stone_center[1])
	y_minz, y_maxz, x_minz, x_maxz = detect_bounding_box(z_proj, seedpoint_z, "proj_z.png" if debug_level>=DebugLevel.SAVE_IMAGES.value else None)
	# print(f'Seedpoint: {seedpoint_z}')
	# print(f'stone bounding box (z): {x_minz}:{x_maxz}, {y_minz}:{y_maxz}')

	# get the smallest bounding box that contains all the stone projections
	x_min = np.max([x_miny, x_minz]); x_max = np.min([x_maxy, x_maxz])
	y_min = np.max([y_minx, y_minz]); y_max = np.min([y_maxx, y_maxz])
	z_min = np.max([z_minx, z_miny]); z_max = np.min([z_maxx, z_maxy])

	# print(f"stone bounding box:")
	# print(f"X: {x_min}:{x_max}")
	# print(f"Y: {y_min}:{y_max}")
	# print(f"Z: {z_min}:{z_max}")  
	return np.array([x_min, x_max, y_min, y_max, z_min, z_max], dtype=int)


def find_glue(volume, stone_value, lims, h, debug_level=DebugLevel.NO_VERBOSE.value, dcm_convention=True):
	# The glue is below the stone and usually is darker than the stone
	# The following code will start looking for glue above the stone
	# and will detect the glue based on the its intensity.
	# It will look below the stone if the glue is not detected above the stone
	glue_pos = None
	is_glue_below = None

	for _, looking_below in enumerate([True, False]):
		if is_glue_below is None:
			logging.info(f"Looking below: {looking_below}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None
			if looking_below ^ (not dcm_convention): # if the image is in DCM convention, the z axis is inverted
				h_clipped = min(h, volume.shape[2]-lims[5])
				bool_vol = volume[lims[0]:lims[1], lims[2]:lims[3], lims[5]:lims[5]+h_clipped] > stone_value*0.2
				new_vol = np.where(bool_vol, volume[lims[0]:lims[1], lims[2]:lims[3], lims[5]:lims[5]+h_clipped], 0)
			else:
				h_clipped = min(h, lims[4])
				bool_vol = volume[lims[0]:lims[1], lims[2]:lims[3], lims[4]-h_clipped:lims[4]] > stone_value*0.2
				new_vol = np.where(bool_vol, volume[lims[0]:lims[1], lims[2]:lims[3], lims[4]-h_clipped:lims[4]], 0)

			proj_x = np.sum(new_vol, axis=0)/(np.sum(bool_vol, axis=0)+1e-6)
			proj_y = np.sum(new_vol, axis=1)/(np.sum(bool_vol, axis=1)+1e-6)
			
			# look at each row in the projection and detect the glue
			cum_x = np.cumsum(proj_x, axis=0); cum_x = cum_x/(cum_x[-1]+1e-6)
			cum_y = np.cumsum(proj_y, axis=0); cum_y = cum_y/(cum_y[-1]+1e-6)

			occurrences_x = np.argmax(cum_x > 0.5, axis=0)+lims[2]
			occurrences_y = np.argmax(cum_y > 0.5, axis=0)+lims[0]
			logging.info(f"occurrences_x shape: {occurrences_x.shape}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None
			logging.info(f"occurrences_y shape: {occurrences_y.shape}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None


			# glue should be in the middle of the volume
			#if occurrences_x.mean() > 0.3 and occurrences_x.mean() < 0.7 and occurrences_y.mean() > 0.3 and occurrences_y.mean() < 0.7:
			if (occurrences_x.shape[0]!=0 and occurrences_y.shape[0]!=0 and np.all(occurrences_x>0.3*volume.shape[0]) and np.all(occurrences_x<0.7*volume.shape[0])) and (np.all(occurrences_y>0.3*volume.shape[1]) and np.all(occurrences_y<0.7*volume.shape[1])):
				x_glue = occurrences_y[occurrences_y.shape[0]//2]#+lims[0]
				#x_glue = volume.shape[0]-x_glue
				y_glue = occurrences_x[occurrences_x.shape[0]//2]#+lims[2]
				#y_glue = volume.shape[1]-y_glue
				z_glue = (lims[5] + h_clipped//2) if looking_below^(not dcm_convention) else (lims[4] - h_clipped//2)

				glue_pos = np.array([x_glue, y_glue, z_glue], dtype=int)
				glue_value = np.median(new_vol[bool_vol])
				is_glue_below = looking_below
				logging.info(f"Glue detected")
				logging.info(f"Glue at position: {glue_pos}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None
				logging.info(f'dcm_convention: {dcm_convention}') if debug_level>=DebugLevel.TEXT_VERBOSE.value else None
			else:
				logging.info("Glue not detected")
				glue_value = None
				glue_pos = None

		if debug_level>=DebugLevel.SAVE_IMAGES.value:
			plt.figure()
			proj_x = np.sum(volume, axis=0)
			plt.imshow(proj_x, cmap='gray')
			plt.colorbar()
			if glue_pos is not None:
				plt.plot(glue_pos[2], glue_pos[1], 'ro')
				plt.savefig("proj-x_glue.png")
				plt.close()
			plt.figure()
			proj_y = np.sum(volume, axis=1)
			plt.imshow(proj_y, cmap='gray')
			plt.colorbar()
			if glue_pos is not None:
				plt.plot(glue_pos[2], glue_pos[0], 'ro')
				plt.savefig("proj-y_glue.png")
				plt.close()
	return glue_value, glue_pos, is_glue_below

def find_stone_value(volume, z_value=None, debug_level=DebugLevel.NO_VERBOSE.value):
	if z_value is None:
		z_value = volume.shape[2]//2
	else:
		z_value = z_value
	slice_vol = volume[:,:,z_value]
	max_val = np.max(slice_vol)
	non_zero_values = slice_vol[slice_vol>0.5*max_val]
	stone_value = np.median(non_zero_values)
	# divide the slice in many quadrants and check the mean value in each of them. 
	# Choose the one with the lowest standard deviation, if the mean is close to the median of the non-zero values
	# get its center
	n_divisions = 40
	quadrant_size = [slice_vol.shape[0]//n_divisions, slice_vol.shape[1]//n_divisions]
	quadrants = []
	for i in range(n_divisions):
		for j in range(n_divisions):
			quadrants.append(slice_vol[i*quadrant_size[0]:(i+1)*quadrant_size[0], j*quadrant_size[1]:(j+1)*quadrant_size[1]])
	quadrants = np.array(quadrants)
	# calculate the mean and std of each quadrant
	quadrants_mean = np.mean(quadrants, axis=(1,2))
	#quadrants_above_threshold = quadrants_mean>0.3*max_val
	quadrants_std = np.std(quadrants, axis=(1,2))
	#quadrants_mean = quadrants_mean[quadrants_above_threshold]
	#quadrants_std = quadrants_std[quadrants_above_threshold]
	quadrants_score = quadrants_mean - 0.3*quadrants_mean*(quadrants_std/np.max(quadrants_std))
	# get the quadrant with the highest score
	best_quadrant = np.argmax(quadrants_score)
	# get the center of the quadrant
	quadrant_center = [best_quadrant//n_divisions, best_quadrant%n_divisions]
	quadrant_center = [quadrant_center[0]*quadrant_size[0]+quadrant_size[0]//2, quadrant_center[1]*quadrant_size[1]+quadrant_size[1]//2]
	if debug_level>=DebugLevel.SAVE_IMAGES.value:
		plt.imshow(slice_vol)
		plt.plot(quadrant_center[1], quadrant_center[0], 'ro')
		plt.savefig("quadrants.png")
		plt.close()
	stone_center = [quadrant_center[0], quadrant_center[1], z_value]
	return stone_value, stone_center
	
def check_identity_vtk_images(image1,image2):
	are_equal = check_metadata_vtk_images(image1,image2)
	array1 = numpy_support.vtk_to_numpy(image1.GetPointData().GetScalars())
	array2 = numpy_support.vtk_to_numpy(image2.GetPointData().GetScalars())
	if not np.array_equal(array1, array2):
		print("Images are different")
		print(f'First 10 elements of array1: {array1[:10]}')
		print(f'First 10 elements of array2: {array2[:10]}')
		are_equal = False
	return are_equal

def check_metadata_vtk_images(image1,image2):
	are_equal = True
	if image1.GetOrigin() != image2.GetOrigin():
		print(f"Origin: {image1.GetOrigin()} != {image2.GetOrigin()}")
		are_equal = False
	if image1.GetSpacing() != image2.GetSpacing():
		print(f"Spacing: {image1.GetSpacing()} != {image2.GetSpacing()}")
		are_equal = False
	if image1.GetDirectionMatrix() != image2.GetDirectionMatrix():
		are_equal = False
	if image1.GetExtent() != image2.GetExtent():
		print(f"Extent: {image1.GetExtent()} != {image2.GetExtent()}")
		are_equal = False
	return are_equal

def vtk_to_sitk(vtk_image):
	sitk_image = sitk.GetImageFromArray(numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars()).reshape(vtk_image.GetDimensions(),order='F').transpose(2,1,0))
	sitk_image.SetOrigin(vtk_image.GetOrigin())
	sitk_image.SetSpacing(vtk_image.GetSpacing())
	sitk_image.SetDirection(vtk_image.GetDirectionMatrix().GetData())
	extent = vtk_image.GetExtent()
	dirs = vtk.vtkMatrix4x4()
	vtk_image.GetImageToWorldMatrix(dirs)
	return sitk_image, extent, dirs

def sitk_to_vtk(sitk_image, extent=None, directions=None):
	vtk_image = vtkSegmentationCore.vtkOrientedImageData()
	vtk_image.SetDimensions(sitk_image.GetSize())
	dtype = sitk_image.GetPixelIDValue()
	if dtype == sitk.sitkUInt8:
		vtk_data_type = vtk.VTK_UNSIGNED_CHAR
	elif dtype == sitk.sitkUInt16:
		vtk_data_type = vtk.VTK_UNSIGNED_SHORT
	elif dtype == sitk.sitkFloat32:
		vtk_data_type = vtk.VTK_FLOAT
	elif dtype == sitk.sitkFloat64:
		vtk_data_type = vtk.VTK_DOUBLE
	else:
		raise ValueError(f"Unsupported pixel type: {dtype}")
	vtk_image.AllocateScalars(vtk_data_type, 1)
	vtk_image.GetPointData().GetScalars().DeepCopy(numpy_support.numpy_to_vtk(sitk.GetArrayFromImage(sitk_image).transpose(2,1,0).ravel(order='F'), deep=True, array_type=vtk_data_type))
	vtk_image.SetOrigin(sitk_image.GetOrigin())
	vtk_image.SetSpacing(sitk_image.GetSpacing())
	if extent is not None:
		vtk_image.SetExtent(extent)
	vtk_image.GetPointData().GetScalars().SetName('ImageScalars')
	direction = sitk_image.GetDirection()
	matrix = vtk.vtkMatrix4x4()
	for i in range(3):
		for j in range(3):
			matrix.SetElement(i, j, direction[i*3+j])
	vtk_image.SetDirectionMatrix(matrix)
	if directions is not None:
		vtk_image.SetImageToWorldMatrix(directions)
	return vtk_image

def apply_curvature_flow(image, dataset_path=None, result=None, debug_level=DebugLevel.NO_VERBOSE.value, **args):
	"""
	It breaks the vtk flow as a conversion to sitk is needed.
	Dataset path is used to save the image before and after the smoothing, for debugging purposes.
	"""
	sitk_image, extent, dirs = vtk_to_sitk(image)
	if debug_level>=DebugLevel.SAVE_IMAGES.value:
		sitk.WriteImage(sitk_image, dataset_path + '_before-smooth.tif') if dataset_path is not None else None
	smoothed_sitk_image = sitk.MinMaxCurvatureFlow(sitk_image, **args)
	if debug_level>=DebugLevel.SAVE_IMAGES.value:
		sitk.WriteImage(smoothed_sitk_image, dataset_path + '_past-smooth.tif') if dataset_path is not None else None
	if result is None:
		return sitk_to_vtk(smoothed_sitk_image, extent, dirs)
	else:
		result[0] = sitk_to_vtk(smoothed_sitk_image, extent, dirs)

def create_initial_seed_segmentation(masterVolumeNode, stone_seed_points, lims, glue_pos, h, seed_radius=10, is_glue_below=True, debug_level=DebugLevel.NO_VERBOSE.value, dcm_convention=True):
	segmentationNode = slicer.vtkMRMLSegmentationNode()
	segmentationNode.SetName("Initial seed")
	if debug_level>=DebugLevel.SHOW_INTERM_SEGMS.value:
		slicer.mrmlScene.AddNode(segmentationNode)
		segmentationNode.CreateDefaultDisplayNodes() # only needed for display
	segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
	dirs = vtk.vtkMatrix4x4()
	masterVolumeNode.GetIJKToRASMatrix(dirs)
	voxel_size = masterVolumeNode.GetSpacing()
	logging.error(f"Voxel size: {voxel_size}") ############## FIXME
	max_vox_size = max(voxel_size)
	origin = masterVolumeNode.GetOrigin()

	# Create seed segment outside stone
	logging.info(f"Background seed positions / X: {lims[0]},{lims[1]} / Y: {lims[2]},{lims[3]} / Z: {lims[4]},{lims[5]}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None
	backgroundSeedPositions = [
		[lims[0], lims[2], lims[4]],
		[lims[1], lims[2], lims[4]],
		[lims[0], lims[3], lims[4]],
		[lims[1], lims[3], lims[4]],
		[lims[0], lims[2], lims[5]],
		[lims[1], lims[2], lims[5]],
		[lims[0], lims[3], lims[5]],
		[lims[1], lims[3], lims[5]]
	]
	bg_append = vtk.vtkAppendPolyData()
	for backgroundSeedPosition in backgroundSeedPositions:
		backgroundSeed = vtk.vtkSphereSource()
		backgroundSeed.SetCenter(origin[0]+dirs.GetElement(0, 0)*backgroundSeedPosition[0], origin[1]+dirs.GetElement(1, 1)*backgroundSeedPosition[1], origin[2]+dirs.GetElement(2, 2)*backgroundSeedPosition[2])
		logging.info(f"Background seed position ({backgroundSeedPosition}): {backgroundSeed.GetCenter()}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None
		backgroundSeed.SetRadius(seed_radius*max_vox_size)
		backgroundSeed.Update()
		bg_append.AddInputData(backgroundSeed.GetOutput())
	bg_append.Update()

	# Create seed segment inside stone
	# stone_seed = vtk.vtkSphereSource()
	# stone_seed.SetCenter(origin[0]+dirs.GetElement(0, 0)*stone_center[0], origin[1]+dirs.GetElement(1, 1)*stone_center[1], origin[2]+dirs.GetElement(2, 2)*stone_center[2])
	# stone_seed.SetRadius(seed_radius*max_vox_size)
	# stone_seed.Update()
	stone_append = vtk.vtkAppendPolyData()
	for stone_seed_point in stone_seed_points:
		stone_seed = vtk.vtkSphereSource()
		stone_seed.SetCenter(origin[0]+dirs.GetElement(0, 0)*stone_seed_point[0], origin[1]+dirs.GetElement(1, 1)*stone_seed_point[1], origin[2]+dirs.GetElement(2, 2)*stone_seed_point[2])
		logging.info(f"Stone seed position ({stone_seed_point}): {stone_seed.GetCenter()}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None
		stone_seed.SetRadius(seed_radius*max_vox_size)
		stone_seed.Update()
		stone_append.AddInputData(stone_seed.GetOutput())
	stone_append.Update()

	# Create seed segment inside glue
	if glue_pos is not None:
		glue_seed = vtk.vtkSphereSource()
		glue_seed.SetCenter(origin[0]+dirs.GetElement(0, 0)*glue_pos[0], origin[1]+dirs.GetElement(1, 1)*glue_pos[1], origin[2]+dirs.GetElement(2, 2)*glue_pos[2])
		l1 = abs(glue_pos[2]-lims[5]) if is_glue_below^(not dcm_convention) else abs(glue_pos[2]-lims[4])
		if seed_radius<l1:
			glue_seed.SetRadius(seed_radius*max_vox_size)
		else:
			logging.warning("Glue radius too big. Clipping to h/2")
			print(f"Glue radius too big. Clipping to h/2 ({(l1-1)})")
			glue_seed.SetRadius((l1-1)*max_vox_size)
		glue_seed.Update()
		# Add first the glue seed, then the others as they will have different priorities in grow-cut

	segmentationNode.AddSegmentFromClosedSurfaceRepresentation(bg_append.GetOutput(), "Background", colours["Background"])
	segmentationNode.AddSegmentFromClosedSurfaceRepresentation(glue_seed.GetOutput(), "Glue", colours["Glue"]) if glue_pos is not None else None
	#segmentationNode.AddSegmentFromClosedSurfaceRepresentation(stone_seed.GetOutput(), "Stone", colours["Stone"])
	segmentationNode.AddSegmentFromClosedSurfaceRepresentation(stone_append.GetOutput(), "Stone", colours["Stone"])
	return segmentationNode

def infer_minimal_label_extent(masterVolumeNode, segmentationNode, debug_level=DebugLevel.NO_VERBOSE.value):
	"""
		Infer the extent of the labels to be processed to reduce computation time
	"""
	# Get segment IDs to be processed (we will process all)
	selectedSegmentIds = vtk.vtkStringArray()
	segmentationNode.GetSegmentation().GetSegmentIDs(selectedSegmentIds)

	# Get merged labelmap extent
	mergedLabelmapGeometryImage = vtkSegmentationCore.vtkOrientedImageData()
	segmentationNode.GetSegmentation().SetImageGeometryFromCommonLabelmapGeometry(mergedLabelmapGeometryImage, selectedSegmentIds)
	labelsEffectiveExtent = mergedLabelmapGeometryImage.GetExtent()
	print(f"Labels effective extent: {labelsEffectiveExtent}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None

	# Compute extent that will be passed to the algorithm (labels effective extent slightly expanded).
	# We assume that the master volume has the same origin, spacing, etc. as the segmentation.
	masterImageData =  vtkSlicerSegmentationsModuleLogic.vtkSlicerSegmentationsModuleLogic.CreateOrientedImageDataFromVolumeNode(masterVolumeNode)
	masterImageData.UnRegister(None)
	masterImageExtent = masterImageData.GetExtent()
	print(f"Master image extent: {masterImageExtent}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None
	margin = [6, 6, 6]
	labelsExpandedExtent = [
		max(masterImageExtent[0], labelsEffectiveExtent[0]-margin[0]),
		min(masterImageExtent[1], labelsEffectiveExtent[1]+margin[0]),
		max(masterImageExtent[2], labelsEffectiveExtent[2]-margin[1]),
		min(masterImageExtent[3], labelsEffectiveExtent[3]+margin[1]),
		max(masterImageExtent[4], labelsEffectiveExtent[4]-margin[2]),
		min(masterImageExtent[5], labelsEffectiveExtent[5]+margin[2]) ]
	#round to positive integers
	mergedLabelmapGeometryImage.SetExtent(labelsExpandedExtent)

	# Create merged labelmap input for growcut
	mergedImage = vtkSegmentationCore.vtkOrientedImageData()
	segmentationNode.GenerateMergedLabelmapForAllSegments(mergedImage,
		vtkSegmentationCore.vtkSegmentation.EXTENT_UNION_OF_EFFECTIVE_SEGMENTS, mergedLabelmapGeometryImage, selectedSegmentIds)
	# the last might be needed to get to update the segments in the segmentation node
	return masterImageData, mergedImage, mergedLabelmapGeometryImage, selectedSegmentIds

def downscale(inputFilter, dims, interpolation_mode='nearest'):
	if inputFilter.GetDimensions() == dims:
		return inputFilter
	else:
		Downsampler = vtk.vtkImageResize()
		Downsampler.SetInputData(inputFilter)
		Downsampler.SetOutputDimensions(dims)
		interpolator = vtk.vtkImageInterpolator()
		if interpolation_mode == 'nearest':
			interpolator.SetInterpolationModeToNearest()
		else:
			interpolator.SetInterpolationModeToLinear()
		Downsampler.SetInterpolator(interpolator)
		Downsampler.Update()
		return Downsampler.GetOutput()

def upscale(inputFilter, dims, interpolation_mode='nearest'):
	if inputFilter.GetDimensions() == dims:
		return inputFilter
	else:
		Upsampler = vtk.vtkImageResize()
		Upsampler.SetInputData(inputFilter)
		Upsampler.SetOutputDimensions(dims)
		interpolator = vtk.vtkImageInterpolator()
		if interpolation_mode == 'nearest':
			interpolator.SetInterpolationModeToNearest()
		else:
			interpolator.SetInterpolationModeToLinear()
		Upsampler.SetInterpolator(interpolator)
		Upsampler.Update()
		return Upsampler.GetOutput()

def vtkInvertMask(image):
	# Subtract 1 to the dilated image and invert the sign
	math = vtk.vtkImageMathematics()
	math.SetInput1Data(image)
	math.SetOperationToAddConstant()
	math.SetConstantC(-1)
	math.Update()

	# Invert the sign
	invert = vtk.vtkImageMathematics()
	invert.SetInput1Data(math.GetOutput())
	invert.SetOperationToMultiplyByK()
	invert.SetConstantK(-1)
	invert.Update()

	return invert

def vtkDilate(image, dilate_l):
	dilate = vtk.vtkImageDilateErode3D()
	dilate.SetInputData(image)
	dilate.SetDilateValue(1)
	dilate.SetErodeValue(0)
	dilate.SetKernelSize(dilate_l, dilate_l, dilate_l)
	dilate.Update()
	return dilate.GetOutput()

def vtkErode(image, erode_l):
	erode = vtk.vtkImageDilateErode3D()
	erode.SetInputData(image)
	erode.SetDilateValue(0)
	erode.SetErodeValue(1)
	erode.SetKernelSize(erode_l, erode_l, erode_l)
	erode.Update()
	return erode.GetOutput()

def clip_image(image, extent):
	"""
	Clip master image data to a smaller extent to reduce computation time
	"""
	clipper = vtk.vtkImageConstantPad()
	clipper.SetInputData(image)
	clipper.SetOutputWholeExtent(extent)
	clipper.Update()
	clipped = vtkSegmentationCore.vtkOrientedImageData()
	clipped.DeepCopy(clipper.GetOutput())
	clipped.CopyDirections(image)
	return clipped

def perform_grow_cut(image, seedlabel, mask=None):
	"""
	Perform grow-cut segmentation
	"""
	fastgrowcut = vtkITK.vtkITKGrowCut()
	fastgrowcut.SetIntensityVolume(image)
	fastgrowcut.SetSeedLabelVolume(seedlabel)
	if mask is not None:
		fastgrowcut.SetMaskVolume(mask)
	fastgrowcut.Update()
	
	fastgrowcutted = vtkSegmentationCore.vtkOrientedImageData()
	fastgrowcutted.DeepCopy(fastgrowcut.GetOutput())
	fastgrowcutted.CopyDirections(image)
	return fastgrowcutted

def addNewSegmentNode(image, geometry, node_name, image_name, color):
	segmentationNode = slicer.vtkMRMLSegmentationNode()
	segmentationNode.SetName(node_name)
	slicer.mrmlScene.AddNode(segmentationNode)
	orientedImageData = vtkSegmentationCore.vtkOrientedImageData()
	orientedImageData.ShallowCopy(image)
	orientedImageData.CopyDirections(geometry)
	segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(orientedImageData, image_name, color)
	return segmentationNode

def updateSegmentNode(segmentationNode, image, geometry, image_name, color):
	orientedImageData = vtkSegmentationCore.vtkOrientedImageData()
	orientedImageData.ShallowCopy(image)
	orientedImageData.CopyDirections(geometry)
	
	name_id = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(image_name)
	if name_id == "": 
		slicer.mrmlScene.AddNode(segmentationNode)
		segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(orientedImageData, image_name, color)
	else:
		slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(orientedImageData, segmentationNode, name_id)

def addNewSegmentationNode(image, geometry, segmentation_name, names, colours_to_use):
	"""
	Create a new segment for the result, based on the input segmentation node for the label names and colours
	"""
	# Create a new segmentation node if it does not exist another with the same name (getFirstNodeByClassByName)
	segmentationNodeFinal = slicer.util.getFirstNodeByClassByName("vtkMRMLSegmentationNode", segmentation_name)
	if not segmentationNodeFinal:
		segmentationNodeFinal = slicer.vtkMRMLSegmentationNode()
		segmentationNodeFinal.SetName(segmentation_name)
		slicer.mrmlScene.AddNode(segmentationNodeFinal)
	resultImage = vtkSegmentationCore.vtkOrientedImageData()
	resultImage.ShallowCopy(image)
	resultImage.CopyDirections(geometry)
	selectedSegmentIds = vtk.vtkStringArray()
	# add the segments to selectedSegmentId
	selectedSegmentIds.SetNumberOfValues(len(names))
	for i, name in enumerate(names):
		selectedSegmentIds.SetValue(i, name)
	slicer.vtkSlicerSegmentationsModuleLogic.ImportLabelmapToSegmentationNode(resultImage, segmentationNodeFinal, selectedSegmentIds)
	# Set the colours
	for i, name in enumerate(names):
		segmentationNodeFinal.GetSegmentation().GetSegment(name).SetColor(colours_to_use[i])
	return segmentationNodeFinal

def updateSegmentationNode(segmentationNode, image, geometry, names, colours_to_use):
	"""
	Update the segments of an existing segmentation node or create them if they do not exist
	"""
	resultImage = vtkSegmentationCore.vtkOrientedImageData()
	resultImage.ShallowCopy(image)
	resultImage.CopyDirections(geometry)

	existingSegmentIds = vtk.vtkStringArray()
	segmentationNode.GetSegmentation().GetSegmentIDs(existingSegmentIds)
	existingSegment_ids = [existingSegmentIds.GetValue(i) for i in range(existingSegmentIds.GetNumberOfValues())]

	selectedSegmentIds = vtk.vtkStringArray()
	selectedSegmentIds.SetNumberOfValues(len(names))
	for i, name in enumerate(names):
		selectedSegmentIds.SetValue(i, name)
		if name in existingSegment_ids:
			# print(f"Segment {name} already exists")	
			# erase the segment
			segmentationNode.GetSegmentation().RemoveSegment(name)
	slicer.vtkSlicerSegmentationsModuleLogic.ImportLabelmapToSegmentationNode(resultImage, segmentationNode, selectedSegmentIds)
	# Set the colours
	for i, name in enumerate(names):
		segmentationNode.GetSegmentation().GetSegment(name).SetColor(colours_to_use[i])

def getStoneLabelFromResult(result, glue_value):
	'''
	Select the stone result by thresholding
	'''
	threshold = vtk.vtkImageThreshold()
	threshold.SetInputData(result)
	threshold.ThresholdByLower(2 if glue_value is not None else 1) # select just the stone
	threshold.SetInValue(0)
	threshold.SetOutValue(1)
	threshold.Update()

	stoneImage = vtkSegmentationCore.vtkOrientedImageData()
	stoneImage.DeepCopy(threshold.GetOutput())
	stoneImage.CopyDirections(result)

	return stoneImage

def getStoneBorder(image):
	'''
	Estimate the gradient of the Stone label. This is the border of the stone.

	'''
	sitk_image, extent, directions = vtk_to_sitk(image)

	# Apply the gradient magnitude filter
	gradientFilter = sitk.GradientMagnitudeImageFilter()
	gradients = gradientFilter.Execute(sitk_image)

	gradients = sitk_to_vtk(gradients, extent, directions)
	
	#fixme: check min/max
	#max_val = np.max(numpy_support.vtk_to_numpy(gradients.GetPointData().GetScalars()))

	gradients = binarise_image(gradients, threshold=0.1*1/image.GetSpacing()[0]) # 10% of the inverse of the voxel size, as the gradient is calculated on a binarised image
	return gradients

def binarise_image(image, threshold=0.5):
	thresholder = vtk.vtkImageThreshold()
	thresholder.SetInputData(image)
	thresholder.ThresholdByLower(threshold)
	thresholder.SetInValue(0)
	thresholder.SetOutValue(1)
	thresholder.Update()
	return thresholder.GetOutput()

def stitch_masked_growcut(labelmap, mask, growcut_result):
	'''
	Add the result of growcut to the initial labelmap. 
	Avoid overlapping of masked region, as the labelmap is also within the mask so that growcut knows what has to grow
	'''
	remover = vtk.vtkImageMathematics()
	remover.SetInput1Data(labelmap)
	remover.SetInput2Data(mask)
	remover.SetOperationToMultiply()
	remover.Update()

	summer = vtk.vtkImageMathematics()
	summer.SetInput1Data(remover.GetOutput())
	summer.SetInput2Data(growcut_result)
	summer.SetOperationToAdd()
	summer.Update()

	return summer.GetOutput()

def getNewGrowcutSeedAndMask(fastgrowcutResult, glue_value, dilate_l, erode_l, debug_level=0, debug_kwargs={}):
	"""
		Extract the border of the stone and dilate it to create a mask for the new growcut
	"""
		
	stone_growcut = getStoneLabelFromResult(fastgrowcutResult, glue_value)
	stone_border = getStoneBorder(stone_growcut)

	# maybe not needed
	# directions = vtk.vtkMatrix4x4()
	# masterVolumeNode.GetIJKToRASMatrix(directions)
	# gradients.SetDirectionMatrix(directions)

	dilated_borders = vtkDilate(stone_border, dilate_l)

	if debug_level >= DebugLevel.SHOW_INTERM_SEGMS.value:
		debug_node = addNewSegmentNode( dilated_borders, debug_kwargs['mergedLabelmapGeometryImage'], "Debug", "Where to update", colours["Border"])
		debug_node.CreateDefaultDisplayNodes() # only needed for display
		debug_node.SetReferenceImageGeometryParameterFromVolumeNode(debug_kwargs['masterVolumeNode'])

	invert = vtkInvertMask(dilated_borders)

	# cast to the same type as the fastgrowcut result
	casted_invert = vtk.vtkImageCast()
	casted_invert.SetInputData(invert.GetOutput())
	casted_invert.SetOutputScalarType(fastgrowcutResult.GetScalarType())
	casted_invert.Update()

	# Set fastgrowcutResult at 0 in the masked region
	cleaned_growseed = vtk.vtkImageMathematics()
	cleaned_growseed.SetInput1Data(fastgrowcutResult)
	cleaned_growseed.SetInput2Data(casted_invert.GetOutput())
	cleaned_growseed.SetOperationToMultiply()
	cleaned_growseed.Update()

	# erode the mask
	border_mask = vtkErode(casted_invert.GetOutput(), erode_l)

	if debug_level >= DebugLevel.SHOW_INTERM_SEGMS.value:
		debug_node = addNewSegmentNode( border_mask, debug_kwargs['mergedLabelmapGeometryImage'], "Debug", "Outside evaluation", colours["Border"])
		debug_node.CreateDefaultDisplayNodes() # only needed for display
		debug_node.SetReferenceImageGeometryParameterFromVolumeNode(debug_kwargs['masterVolumeNode'])

	return cleaned_growseed.GetOutput(), border_mask

def checkInputs(**kwargs):
	sanitised_kwargs = kwargs.copy()

	return sanitised_kwargs

def MeasureStats(node):
	import SegmentStatistics
	segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
	segStatLogic.getParameterNode().SetParameter("Segmentation", node.GetID())
	segStatLogic.computeStatistics()
	stats = segStatLogic.getStatistics()
	# Display volume and surface area of each segment
	for segmentId in stats["SegmentIDs"]:
		volume_mm3 = stats[segmentId,"LabelmapSegmentStatisticsPlugin.volume_mm3"]
		segmentName = node.GetSegmentation().GetSegment(segmentId).GetName()
		#print(f"{segmentName} volume = {volume_mm3} mm3")
		logging.info(f"{volume_mm3*0.01757625:.2f} Ct, stone weight\n\n")
		

def segmentation_kernel(**kwargs):
		
	masterVolumeNode = slicer.mrmlScene.GetNodeByID(kwargs['inputVolume'])
	outputSegmentationNode = slicer.mrmlScene.GetNodeByID(kwargs['outputLabelMap'])



	# Check the orientation of the volume to know if the z axis is inverted.
	orientation = vtk.vtkMatrix4x4()
	masterVolumeNode.GetIJKToRASMatrix(orientation)
	dcm_convention = (orientation.GetElement(2, 2) < 0) 
	del orientation

	"""
		Parameters
	"""

	kwargs = checkInputs(**kwargs)
	
	# scale down the image for faster computation. Must be bigger than 1	
	scale = kwargs['scale']
	# parameters for the curvature flow
	want_curvature_flow = kwargs['want_curvature_flow']
	curvature_timestep = kwargs['curvature_timestep']
	curvature_niter = kwargs['curvature_niter']
	curvature_args = {'timeStep':curvature_timestep, 'numberOfIterations':curvature_niter}
	# define where the top seed point of the stone will be
	top_stone_seed_perc = kwargs['top_stone_seed_perc']
	# define where the bottom seed point of the stone will be
	bottom_stone_seed_perc = kwargs['bottom_stone_seed_perc']

	# define where the second growcut will be executed around the stone border
	border_dilate_radius = kwargs['border_dilate_radius']
	# erode dilated stone border. This will create a smaller region where label values will be reassigned by the second growcut
	border_erode_radius = kwargs['border_erode_radius']
	# look just at h voxels below the stone for the glue
	glue_h = kwargs['glue_h']
	# radius of the seeds for the growcut
	seed_radius = kwargs['seed_radius']
	# Advanced parameters
	debug_level = kwargs['debug_level']

	"""
		Beginning of the code
	"""
	debug = DebugLevel(debug_level)

	timings = {}
	timings["Total"] = -time.time()

	volume = slicer.util.arrayFromVolume(masterVolumeNode)
	volume = volume.transpose(2,1,0)
	min_volume = np.min(volume)
	volume = volume - min_volume

	timings["Find stone"] = -time.time()
	stone_value, stone_center = find_stone_value(volume, debug_level=debug_level)
	timings["Find stone"] += time.time()
	logging.info(f"stone gray value: {stone_value + min_volume}") if debug_level>=DebugLevel.TEXT_VERBOSE.value else None

	timings["Find minimal bounding box"] = -time.time()
	stone_lims = find_minimal_bounding_box(volume, stone_center, stone_value, debug_level=debug_level)
	timings["Find minimal bounding box"] += time.time()

	timings["Find glue"] = -time.time()
	glue_value, glue_pos, is_glue_below = find_glue(volume, stone_value, stone_lims, glue_h, debug_level=debug_level, dcm_convention=dcm_convention)
	timings["Find glue"] += time.time()
	logging.info(f"Glue gray value: {glue_value + min_volume}") if (debug_level>=DebugLevel.TEXT_VERBOSE.value and glue_value is not None) else None

	objects_in_scene = ["Background", "Glue", "Stone"] if glue_pos is not None else ["Background", "Stone"]

	stone_seed_points = []
	stone_seed_points.append(stone_center)
	# find additional seed points for the stone
	_, top_stone_pos = find_stone_value(volume, debug_level=debug_level, z_value=int(top_stone_seed_perc*(stone_lims[5]-stone_lims[4])+stone_lims[4]))
	_, bottom_stone_pos = find_stone_value(volume, debug_level=debug_level, z_value=int(bottom_stone_seed_perc*(stone_lims[5]-stone_lims[4])+stone_lims[4]))
	stone_seed_points.append(top_stone_pos)
	stone_seed_points.append(bottom_stone_pos)

	del volume

	timings["Create seed"] = -time.time()
	segmentationNode = create_initial_seed_segmentation(masterVolumeNode, stone_seed_points, stone_lims, glue_pos, glue_h, seed_radius=seed_radius, is_glue_below=is_glue_below, debug_level=debug_level, dcm_convention=dcm_convention)
	timings["Create seed"] += time.time()

	masterImageData, mergedImage, mergedLabelmapGeometryImage, _ = infer_minimal_label_extent(masterVolumeNode, segmentationNode, debug_level=debug_level)
	masterImageClipped = clip_image(masterImageData, mergedLabelmapGeometryImage.GetExtent())

	if want_curvature_flow:
		# run the apply_curvature_flow in another thread
		results = [None]
		curvature_flow_thread = threading.Thread(target=apply_curvature_flow, kwargs={'image':masterImageClipped, **curvature_args, 'result':results})
		curvature_flow_thread.start()

	timings["Downscale"] = -time.time()
	down_dims = [int(x/scale) for x in masterImageClipped.GetDimensions()]
	masterImageClippedDownsampled = downscale(masterImageClipped, down_dims, interpolation_mode='linear')	
	mergedImageDownsampled = downscale(mergedImage, down_dims, interpolation_mode='nearest')
	timings["Downscale"] += time.time()

	timings["Grow Region"] = -time.time()
	fastgrowcutted = perform_grow_cut(masterImageClippedDownsampled, mergedImageDownsampled, mask=None)
	timings["Grow Region"] += time.time()

	del masterImageClippedDownsampled, mergedImageDownsampled

	timings["Upscale"] = -time.time()
	dims = [ masterImageClipped.GetDimensions()[i] for i in range(3) ]
	fastgrowcutted = upscale(fastgrowcutted, dims, interpolation_mode='nearest')
	timings["Upscale"] += time.time()

	for i in range(3):
		logging.error(f"Something went wrong in the upscaling. Dimension {i} is different. {masterImageClipped.GetDimensions()} != {fastgrowcutted.GetDimensions()}") if not masterImageClipped.GetDimensions()[i] == fastgrowcutted.GetDimensions()[i] else None

	if want_curvature_flow or scale > 1: # in these case it makes sense to apply the growcut again
		if debug_level >= DebugLevel.SHOW_INTERM_SEGMS.value:
			#updateSegmentationNode(segmentationNode, fastgrowcutted, mergedLabelmapGeometryImage)
			intermNode = addNewSegmentationNode(fastgrowcutted, mergedLabelmapGeometryImage, "Intermediate Segmentation", objects_in_scene, [colours[o] for o in objects_in_scene])			
			intermNode.CreateDefaultDisplayNodes() # only needed for display
			intermNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
		slicer.util.resetSliceViews() 

		timings["Get stone border"] = -time.time()
		new_growcut_seed, border_mask = getNewGrowcutSeedAndMask(fastgrowcutted, glue_value, border_dilate_radius, border_erode_radius, debug_level=debug_level, debug_kwargs={'mergedLabelmapGeometryImage':mergedLabelmapGeometryImage, 'masterVolumeNode':masterVolumeNode})
		timings["Get stone border"] += time.time()

		# Use the dilated image as a mask for the a new growcut. do not use the downsampled image
		if want_curvature_flow:
			timings["Curvature flow / wait time"] = -time.time()
			curvature_flow_thread.join()
			timings["Curvature flow / wait time"] += time.time()
			image = results[0]
		else:
			image = masterImageClipped
		timings["Grow Region 2"] = -time.time()
		fastgrowcutted2 = perform_grow_cut(image, new_growcut_seed, mask=border_mask)
		fastgrowcutted2 = stitch_masked_growcut(new_growcut_seed, border_mask, fastgrowcutted2)
		timings["Grow Region 2"] += time.time()
		fastgrowcutted = fastgrowcutted2

	if debug_level < DebugLevel.SHOW_INTERM_SEGMS.value:
		stone_label = getStoneLabelFromResult(fastgrowcutted, glue_value) 
		updateSegmentNode(outputSegmentationNode, stone_label, mergedLabelmapGeometryImage, 'Stone', colours['Stone'])		
		outputSegmentationNode.CreateDefaultDisplayNodes() # only needed for display
		outputSegmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
		outputSegmentationNode.CreateClosedSurfaceRepresentation()
		MeasureStats(outputSegmentationNode)
	else:
		updateSegmentationNode(outputSegmentationNode, fastgrowcutted, mergedLabelmapGeometryImage, objects_in_scene, [colours[o] for o in objects_in_scene])

	timings["Total"] += time.time()


	# Set the output volume path
	inputVolumePath = masterVolumeNode.GetStorageNode().GetFileName()
	path = os.path.dirname(inputVolumePath)
	outputVolumePath = os.path.join(path, "labelmap.nrrd")
	outputSegmentationNode.AddDefaultStorageNode()
	outputSegmentationNode.GetStorageNode().SetFileName(outputVolumePath)



	# print timing
	logging.info("Timings of relevant modules:\n")
	for key, value in timings.items():
		logging.info(f'{value:.2f} s for [{key}]')

