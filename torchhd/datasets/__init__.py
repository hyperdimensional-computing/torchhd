#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from torchhd.datasets.beijing_air_quality import BeijingAirQuality
from torchhd.datasets.isolet import ISOLET
from torchhd.datasets.european_languages import EuropeanLanguages
from torchhd.datasets.ucihar import UCIHAR
from torchhd.datasets.airfoil_self_noise import AirfoilSelfNoise
from torchhd.datasets.emg_hand_gestures import EMGHandGestures
from torchhd.datasets.pamap import PAMAP
from torchhd.datasets.ccpp import CyclePowerPlant
from torchhd.datasets.dataset import CollectionDataset
from torchhd.datasets.dataset import DatasetFourFold
from torchhd.datasets.dataset import DatasetTrainTest
from torchhd.datasets.dataset import UCIClassificationBenchmark
from torchhd.datasets.dataset import HDCArena
from torchhd.datasets.abalone import Abalone
from torchhd.datasets.adult import Adult
from torchhd.datasets.acute_inflammation import AcuteInflammation
from torchhd.datasets.acute_nephritis import AcuteNephritis
from torchhd.datasets.annealing import Annealing
from torchhd.datasets.arrhythmia import Arrhythmia
from torchhd.datasets.audiology_std import AudiologyStd
from torchhd.datasets.balance_scale import BalanceScale
from torchhd.datasets.balloons import Balloons
from torchhd.datasets.bank import Bank
from torchhd.datasets.blood import Blood
from torchhd.datasets.breast_cancer import BreastCancer
from torchhd.datasets.breast_cancer_wisc import BreastCancerWisc
from torchhd.datasets.breast_cancer_wisc_diag import BreastCancerWiscDiag
from torchhd.datasets.breast_cancer_wisc_prog import BreastCancerWiscProg
from torchhd.datasets.breast_tissue import BreastTissue
from torchhd.datasets.car import Car
from torchhd.datasets.cardiotocography_3clases import Cardiotocography3Clases
from torchhd.datasets.cardiotocography_10clases import Cardiotocography10Clases
from torchhd.datasets.chess_krvk import ChessKrvk
from torchhd.datasets.chess_krvkp import ChessKrvkp
from torchhd.datasets.congressional_voting import CongressionalVoting
from torchhd.datasets.conn_bench_sonar_mines_rocks import ConnBenchSonarMinesRocks
from torchhd.datasets.conn_bench_vowel_deterding import ConnBenchVowelDeterding
from torchhd.datasets.connect_4 import Connect4
from torchhd.datasets.contrac import Contrac
from torchhd.datasets.credit_approval import CreditApproval
from torchhd.datasets.cylinder_bands import CylinderBands
from torchhd.datasets.dermatology import Dermatology
from torchhd.datasets.echocardiogram import Echocardiogram
from torchhd.datasets.ecoli import Ecoli
from torchhd.datasets.energy_y1 import EnergyY1
from torchhd.datasets.energy_y2 import EnergyY2
from torchhd.datasets.fertility import Fertility
from torchhd.datasets.flags import Flags
from torchhd.datasets.glass import Glass
from torchhd.datasets.haberman_survival import HabermanSurvival
from torchhd.datasets.hayes_roth import HayesRoth
from torchhd.datasets.heart_cleveland import HeartCleveland
from torchhd.datasets.heart_hungarian import HeartHungarian
from torchhd.datasets.heart_switzerland import HeartSwitzerland
from torchhd.datasets.heart_va import HeartVa
from torchhd.datasets.hepatitis import Hepatitis
from torchhd.datasets.hill_valley import HillValley
from torchhd.datasets.horse_colic import HorseColic
from torchhd.datasets.ilpd_indian_liver import IlpdIndianLiver
from torchhd.datasets.image_segmentation import ImageSegmentation
from torchhd.datasets.ionosphere import Ionosphere
from torchhd.datasets.iris import Iris
from torchhd.datasets.led_display import LedDisplay
from torchhd.datasets.lenses import Lenses
from torchhd.datasets.letter import Letter
from torchhd.datasets.libras import Libras
from torchhd.datasets.low_res_spect import LowResSpect
from torchhd.datasets.lung_cancer import LungCancer
from torchhd.datasets.lymphography import Lymphography
from torchhd.datasets.magic import Magic
from torchhd.datasets.mammographic import Mammographic
from torchhd.datasets.miniboone import Miniboone
from torchhd.datasets.molec_biol_promoter import MolecBiolPromoter
from torchhd.datasets.molec_biol_splice import MolecBiolSplice
from torchhd.datasets.monks_1 import Monks1
from torchhd.datasets.monks_2 import Monks2
from torchhd.datasets.monks_3 import Monks3
from torchhd.datasets.mushroom import Mushroom
from torchhd.datasets.musk_1 import Musk1
from torchhd.datasets.musk_2 import Musk2
from torchhd.datasets.nursery import Nursery
from torchhd.datasets.oocytes_merluccius_nucleus_4d import OocytesMerlucciusNucleus4d
from torchhd.datasets.oocytes_merluccius_states_2f import OocytesMerlucciusStates2f
from torchhd.datasets.oocytes_trisopterus_nucleus_2f import OocytesTrisopterusNucleus2f
from torchhd.datasets.oocytes_trisopterus_states_5b import OocytesTrisopterusStates5b
from torchhd.datasets.optical import Optical
from torchhd.datasets.ozone import Ozone
from torchhd.datasets.page_blocks import PageBlocks
from torchhd.datasets.parkinsons import Parkinsons
from torchhd.datasets.pendigits import Pendigits
from torchhd.datasets.pima import Pima
from torchhd.datasets.pittsburg_bridges_material import PittsburgBridgesMaterial
from torchhd.datasets.pittsburg_bridges_rel_l import PittsburgBridgesRelL
from torchhd.datasets.pittsburg_bridges_span import PittsburgBridgesSpan
from torchhd.datasets.pittsburg_bridges_t_or_d import PittsburgBridgesTOrD
from torchhd.datasets.pittsburg_bridges_type import PittsburgBridgesType
from torchhd.datasets.planning import Planning
from torchhd.datasets.plant_margin import PlantMargin
from torchhd.datasets.plant_shape import PlantShape
from torchhd.datasets.plant_texture import PlantTexture
from torchhd.datasets.post_operative import PostOperative
from torchhd.datasets.primary_tumor import PrimaryTumor
from torchhd.datasets.ringnorm import Ringnorm
from torchhd.datasets.seeds import Seeds
from torchhd.datasets.semeion import Semeion
from torchhd.datasets.soybean import Soybean
from torchhd.datasets.spambase import Spambase
from torchhd.datasets.spect import Spect
from torchhd.datasets.spectf import Spectf
from torchhd.datasets.statlog_australian_credit import StatlogAustralianCredit
from torchhd.datasets.statlog_german_credit import StatlogGermanCredit
from torchhd.datasets.statlog_heart import StatlogHeart
from torchhd.datasets.statlog_image import StatlogImage
from torchhd.datasets.statlog_landsat import StatlogLandsat
from torchhd.datasets.statlog_shuttle import StatlogShuttle
from torchhd.datasets.statlog_vehicle import StatlogVehicle
from torchhd.datasets.steel_plates import SteelPlates
from torchhd.datasets.synthetic_control import SyntheticControl
from torchhd.datasets.teaching import Teaching
from torchhd.datasets.thyroid import Thyroid
from torchhd.datasets.tic_tac_toe import TicTacToe
from torchhd.datasets.titanic import Titanic
from torchhd.datasets.trains import Trains
from torchhd.datasets.twonorm import Twonorm
from torchhd.datasets.vertebral_column_2clases import VertebralColumn2Clases
from torchhd.datasets.vertebral_column_3clases import VertebralColumn3Clases
from torchhd.datasets.wall_following import WallFollowing
from torchhd.datasets.waveform import Waveform
from torchhd.datasets.waveform_noise import WaveformNoise
from torchhd.datasets.wine import Wine
from torchhd.datasets.wine_quality_red import WineQualityRed
from torchhd.datasets.wine_quality_white import WineQualityWhite
from torchhd.datasets.yeast import Yeast
from torchhd.datasets.zoo import Zoo

__all__ = [
    "HDCArena",
    "BeijingAirQuality",
    "ISOLET",
    "EuropeanLanguages",
    "UCIHAR",
    "AirfoilSelfNoise",
    "EMGHandGestures",
    "PAMAP",
    "CyclePowerPlant",
    "CollectionDataset",
    "DatasetFourFold",
    "DatasetTrainTest",
    "UCIClassificationBenchmark",
    "Abalone",
    "Adult",
    "AcuteInflammation",
    "AcuteNephritis",
    "Annealing",
    "Arrhythmia",
    "AudiologyStd",
    "BalanceScale",
    "Balloons",
    "Bank",
    "Blood",
    "BreastCancer",
    "BreastCancerWisc",
    "BreastCancerWiscDiag",
    "BreastCancerWiscProg",
    "BreastTissue",
    "Car",
    "Cardiotocography3Clases",
    "Cardiotocography10Clases",
    "ChessKrvk",
    "ChessKrvkp",
    "CongressionalVoting",
    "ConnBenchSonarMinesRocks",
    "ConnBenchVowelDeterding",
    "Connect4",
    "Contrac",
    "CreditApproval",
    "CylinderBands",
    "Dermatology",
    "Echocardiogram",
    "Ecoli",
    "EnergyY1",
    "EnergyY2",
    "Fertility",
    "Flags",
    "Glass",
    "HabermanSurvival",
    "HayesRoth",
    "HeartCleveland",
    "HeartHungarian",
    "HeartSwitzerland",
    "HeartVa",
    "Hepatitis",
    "HillValley",
    "HorseColic",
    "IlpdIndianLiver",
    "ImageSegmentation",
    "Ionosphere",
    "Iris",
    "LedDisplay",
    "Lenses",
    "Letter",
    "Libras",
    "LowResSpect",
    "LungCancer",
    "Lymphography",
    "Magic",
    "Mammographic",
    "Miniboone",
    "MolecBiolPromoter",
    "MolecBiolSplice",
    "Monks1",
    "Monks2",
    "Monks3",
    "Mushroom",
    "Musk1",
    "Musk2",
    "Nursery",
    "OocytesMerlucciusNucleus4d",
    "OocytesMerlucciusStates2f",
    "OocytesTrisopterusNucleus2f",
    "OocytesTrisopterusStates5b",
    "Optical",
    "Ozone",
    "PageBlocks",
    "Parkinsons",
    "Pendigits",
    "Pima",
    "PittsburgBridgesMaterial",
    "PittsburgBridgesRelL",
    "PittsburgBridgesSpan",
    "PittsburgBridgesTOrD",
    "PittsburgBridgesType",
    "Planning",
    "PlantMargin",
    "PlantShape",
    "PlantTexture",
    "PostOperative",
    "PrimaryTumor",
    "Ringnorm",
    "Seeds",
    "Semeion",
    "Soybean",
    "Spambase",
    "Spect",
    "Spectf",
    "StatlogAustralianCredit",
    "StatlogGermanCredit",
    "StatlogHeart",
    "StatlogImage",
    "StatlogLandsat",
    "StatlogShuttle",
    "StatlogVehicle",
    "SteelPlates",
    "SyntheticControl",
    "Teaching",
    "Thyroid",
    "TicTacToe",
    "Titanic",
    "Trains",
    "Twonorm",
    "VertebralColumn2Clases",
    "VertebralColumn3Clases",
    "WallFollowing",
    "Waveform",
    "WaveformNoise",
    "Wine",
    "WineQualityRed",
    "WineQualityWhite",
    "Yeast",
    "Zoo",
]
