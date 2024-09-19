from tabgpt.data_loader import DataFrameLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


class HousePricesData(DataFrameLoader):
    def __init__(self, task_description='house prices'):
        super().__init__(task_description)
        old_to_new_col_mapping: Dict[str, str] =  None

    def read(self, all_features=True):      
        df_train_full = pd.read_csv(os.path.join(self.current_dir, 'train.csv'))

        df_train_full = self.give_reasonable_names(df_train_full)

        if all_features:
            categorical_features = []
            numerical_features = []
            for col in df_train_full.drop(columns=["Id", "SalePrice"]).columns:
                if df_train_full[col].dtype == 'O':
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)

        else:

            categorical_features = [
                self.old_to_new_col_mapping["OverallQual"],
                self.old_to_new_col_mapping["ExterQual"],
                self.old_to_new_col_mapping["Neighborhood"],
                self.old_to_new_col_mapping["BsmtQual"],
                self.old_to_new_col_mapping["KitchenQual"],
            ]
            numerical_features = [
                self.old_to_new_col_mapping["GarageCars"],
                self.old_to_new_col_mapping["GrLivArea"],
                self.old_to_new_col_mapping["GarageArea"],
                self.old_to_new_col_mapping["TotalBsmtSF"],
                self.old_to_new_col_mapping["YearBuilt"],         
            ]

        return df_train_full, numerical_features, categorical_features


    def setup(self, all_features=True):
        # use data from Kaggle competition https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
        df_train_full, numerical_features, categorical_features = self.read(all_features)
       
        df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=666)
        num_max = df_train[numerical_features].abs().max()
        df_train[numerical_features] = df_train[numerical_features] / num_max
        df_val[numerical_features] = df_val[numerical_features] / num_max
        df_train["target"] = np.log(1 + df_train["SalePrice"])
        df_val["target"] = df_val["SalePrice"]

        self.df_train = df_train
        self.df_val = df_val
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_features = len(numerical_features + categorical_features)
        self.target_column = "target"

    def test_setup(self, all_train_data = False, all_features=True):

        if all_train_data:
            df_train, numerical_features, categorical_features = self.read(all_features)
            num_max = df_train[numerical_features].abs().max()
            df_train[numerical_features] = df_train[numerical_features] / num_max
            df_train["target"] = np.log(1 + df_train["SalePrice"])
            self.df_train = df_train
        else:
            assert self.df_train is not None, 'No training set available, did you call setup() already?'
            numerical_features = self.numerical_features
            num_max = self.df_train[numerical_features].abs().max()
            categorical_features = self.categorical_features

        features = numerical_features + categorical_features
        
        df_test = pd.read_csv(os.path.join(self.current_dir,"test.csv"))
        df_test = self.give_reasonable_names(df_test) 
        df_test = df_test[features + ['Id']]                  
        
        df_test['target'] = df_test['Id']
        df_test[numerical_features] = df_test[numerical_features] / num_max
        df_test['target'] = df_test['Id']
        self.df_test = df_test

    def give_reasonable_names(self, df):
        subclass_mapping = {
            20: '1-STORY 1946 & NEWER ALL STYLES',
            30: '1-STORY 1945 & OLDER',
            40: '1-STORY W/FINISHED ATTIC ALL AGES',
            45: '1-1/2 STORY - UNFINISHED ALL AGES',
            50: '1-1/2 STORY FINISHED ALL AGES',
            60: '2-STORY 1946 & NEWER',
            70: '2-STORY 1945 & OLDER',
            75: '2-1/2 STORY ALL AGES',
            80: 'SPLIT OR MULTI-LEVEL',
            85: 'SPLIT FOYER',
            90: 'DUPLEX - ALL STYLES AND AGES',
            120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
            150: '1-1/2 STORY PUD - ALL AGES',
            160: '2-STORY PUD - 1946 & NEWER',
            180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
            190: '2 FAMILY CONVERSION - ALL STYLES AND AGES'
        }

        mszoning_mapping = {
            'A': 'Agriculture',
            'C': 'Commercial',
            'FV': 'Floating Village Residential',
            'I': 'Industrial',
            'RH': 'Residential High Density',
            'RL': 'Residential Low Density',
            'RP': 'Residential Low Density Park',
            'RM': 'Residential Medium Density'
        }

        street_mapping = {
            'Grvl': 'Gravel',
            'Pave': 'Paved'
        }

        alley_mapping = {
            'Grvl': 'Gravel',
            'Pave': 'Paved',
            'NA': 'No alley access'
        }

        lotshape_mapping = {
            'Reg': 'Regular',
            'IR1': 'Slightly irregular',
            'IR2': 'Moderately Irregular',
            'IR3': 'Irregular'
        }

        landcontour_mapping = {
            'Lvl': 'Near Flat/Level',
            'Bnk': 'Banked - Quick and significant rise from street grade to building',
            'HLS': 'Hillside - Significant slope from side to side',
            'Low': 'Depression'
        }

        utilities_mapping = {
            'AllPub': 'All public Utilities (E,G,W,& S)',
            'NoSewr': 'Electricity, Gas, and Water (Septic Tank)',
            'NoSeWa': 'Electricity and Gas Only',
            'ELO': 'Electricity only'
        }

        lotconfig_mapping = {
            'Inside': 'Inside lot',
            'Corner': 'Corner lot',
            'CulDSac': 'Cul-de-sac',
            'FR2': 'Frontage on 2 sides of property',
            'FR3': 'Frontage on 3 sides of property'
        }

        landslope_mapping = {
            'Gtl': 'Gentle slope',
            'Mod': 'Moderate Slope',
            'Sev': 'Severe Slope'
        }

        neighborhood_mapping = {
            'Blmngtn': 'Bloomington Heights',
            'Blueste': 'Bluestem',
            'BrDale': 'Briardale',
            'BrkSide': 'Brookside',
            'ClearCr': 'Clear Creek',
            'CollgCr': 'College Creek',
            'Crawfor': 'Crawford',
            'Edwards': 'Edwards',
            'Gilbert': 'Gilbert',
            'IDOTRR': 'Iowa DOT and Rail Road',
            'MeadowV': 'Meadow Village',
            'Mitchel': 'Mitchell',
            'Names': 'North Ames',
            'NoRidge': 'Northridge',
            'NPkVill': 'Northpark Villa',
            'NridgHt': 'Northridge Heights',
            'NWAmes': 'Northwest Ames',
            'OldTown': 'Old Town',
            'SWISU': 'South & West of Iowa State University',
            'Sawyer': 'Sawyer',
            'SawyerW': 'Sawyer West',
            'Somerst': 'Somerset',
            'StoneBr': 'Stone Brook',
            'Timber': 'Timberland',
            'Veenker': 'Veenker'
        }

        condition1_mapping = {
            'Artery': 'Adjacent to arterial street',
            'Feedr': 'Adjacent to feeder street',
            'Norm': 'Normal',
            'RRNn': 'Within 200\' of North-South Railroad',
            'RRAn': 'Adjacent to North-South Railroad',
            'PosN': 'Near positive off-site feature--park, greenbelt, etc.',
            'PosA': 'Adjacent to positive off-site feature',
            'RRNe': 'Within 200\' of East-West Railroad',
            'RRAe': 'Adjacent to East-West Railroad'
        }

        condition2_mapping = {
            'Artery': 'Adjacent to arterial street',
            'Feedr': 'Adjacent to feeder street',
            'Norm': 'Normal',
            'RRNn': 'Within 200\' of North-South Railroad',
            'RRAn': 'Adjacent to North-South Railroad',
            'PosN': 'Near positive off-site feature--park, greenbelt, etc.',
            'PosA': 'Adjacent to positive off-site feature',
            'RRNe': 'Within 200\' of East-West Railroad',
            'RRAe': 'Adjacent to East-West Railroad'
        }

        bldgtype_mapping = {
            '1Fam': 'Single-family Detached',
            '2FmCon': 'Two-family Conversion; originally built as one-family dwelling',
            'Duplx': 'Duplex',
            'TwnhsE': 'Townhouse End Unit',
            'TwnhsI': 'Townhouse Inside Unit'
        }

        housestyle_mapping = {
            '1Story': 'One story',
            '1.5Fin': 'One and one-half story: 2nd level finished',
            '1.5Unf': 'One and one-half story: 2nd level unfinished',
            '2Story': 'Two story',
            '2.5Fin': 'Two and one-half story: 2nd level finished',
            '2.5Unf': 'Two and one-half story: 2nd level unfinished',
            'SFoyer': 'Split Foyer',
            'SLvl': 'Split Level'
        }

        overallqual_mapping = {
            10: 'Very Excellent',
            9: 'Excellent',
            8: 'Very Good',
            7: 'Good',
            6: 'Above Average',
            5: 'Average',
            4: 'Below Average',
            3: 'Fair',
            2: 'Poor',
            1: 'Very Poor'
        }

        overallcond_mapping = {
            10: 'Very Excellent',
            9: 'Excellent',
            8: 'Very Good',
            7: 'Good',
            6: 'Above Average',
            5: 'Average',
            4: 'Below Average',
            3: 'Fair',
            2: 'Poor',
            1: 'Very Poor'
        }

        roofstyle_mapping = {
            'Flat': 'Flat',
            'Gable': 'Gable',
            'Gambrel': 'Gabrel (Barn)',
            'Hip': 'Hip',
            'Mansard': 'Mansard',
            'Shed': 'Shed'
        }

        roofmatl_mapping = {
            'ClyTile': 'Clay or Tile',
            'CompShg': 'Standard (Composite) Shingle',
            'Membran': 'Membrane',
            'Metal': 'Metal',
            'Roll': 'Roll',
            'Tar&Grv': 'Gravel & Tar',
            'WdShake': 'Wood Shakes',
            'WdShngl': 'Wood Shingles'
        }

        exterior1st_mapping = {
            'AsbShng': 'Asbestos Shingles',
            'AsphShn': 'Asphalt Shingles',
            'BrkComm': 'Brick Common',
            'BrkFace': 'Brick Face',
            'CBlock': 'Cinder Block',
            'CemntBd': 'Cement Board',
            'HdBoard': 'Hard Board',
            'ImStucc': 'Imitation Stucco',
            'MetalSd': 'Metal Siding',
            'Other': 'Other',
            'Plywood': 'Plywood',
            'PreCast': 'PreCast',
            'Stone': 'Stone',
            'Stucco': 'Stucco',
            'VinylSd': 'Vinyl Siding',
            'Wd Sdng': 'Wood Siding',
            'WdShing': 'Wood Shingles'
        }

        exterior2nd_mapping = {
            'AsbShng': 'Asbestos Shingles',
            'AsphShn': 'Asphalt Shingles',
            'BrkComm': 'Brick Common',
            'BrkFace': 'Brick Face',
            'CBlock': 'Cinder Block',
            'CemntBd': 'Cement Board',
            'HdBoard': 'Hard Board',
            'ImStucc': 'Imitation Stucco',
            'MetalSd': 'Metal Siding',
            'Other': 'Other',
            'Plywood': 'Plywood',
            'PreCast': 'PreCast',
            'Stone': 'Stone',
            'Stucco': 'Stucco',
            'VinylSd': 'Vinyl Siding',
            'Wd Sdng': 'Wood Siding',
            'WdShing': 'Wood Shingles'
        }

        exterqual_mapping = {
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Average/Typical',
            'Fa': 'Fair',
            'Po': 'Poor'
        }

        extercond_mapping = {
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Average/Typical',
            'Fa': 'Fair',
            'Po': 'Poor'
        }

        foundation_mapping = {
            'BrkTil': 'Brick & Tile',
            'CBlock': 'Cinder Block',
            'PConc': 'Poured Concrete',
            'Slab': 'Slab',
            'Stone': 'Stone',
            'Wood': 'Wood'
        }

        bsmtqual_mapping = {
            'Ex': 'Excellent (100+ inches)',
            'Gd': 'Good (90-99 inches)',
            'TA': 'Typical (80-89 inches)',
            'Fa': 'Fair (70-79 inches)',
            'Po': 'Poor (<70 inches)',
            'NA': 'No Basement'
        }

        bsmtcond_mapping = {
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Typical - slight dampness allowed',
            'Fa': 'Fair - dampness or some cracking or settling',
            'Po': 'Poor - Severe cracking, settling, or wetness',
            'NA': 'No Basement'
        }

        bsmtexposure_mapping = {
            'Gd': 'Good Exposure',
            'Av': 'Average Exposure (split levels or foyers typically score average or above)',
            'Mn': 'Minimum Exposure',
            'No': 'No Exposure',
            'NA': 'No Basement'
        }

        bsmtfintype1_mapping = {
            'GLQ': 'Good Living Quarters',
            'ALQ': 'Average Living Quarters',
            'BLQ': 'Below Average Living Quarters',
            'Rec': 'Average Rec Room',
            'LwQ': 'Low Quality',
            'Unf': 'Unfinished',
            'NA': 'No Basement'
        }

        bsmtfintype2_mapping = {
            'GLQ': 'Good Living Quarters',
            'ALQ': 'Average Living Quarters',
            'BLQ': 'Below Average Living Quarters',
            'Rec': 'Average Rec Room',
            'LwQ': 'Low Quality',
            'Unf': 'Unfinished',
            'NA': 'No Basement'
        }

        heating_mapping = {
            'Floor': 'Floor Furnace',
            'GasA': 'Gas forced warm air furnace',
            'GasW': 'Gas hot water or steam heat',
            'Grav': 'Gravity furnace',
            'OthW': 'Hot water or steam heat other than gas',
            'Wall': 'Wall furnace'
        }

        heatingqc_mapping = {
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Average/Typical',
            'Fa': 'Fair',
            'Po': 'Poor'
        }

        centralair_mapping = {
            'N': 'No',
            'Y': 'Yes'
        }

        electrical_mapping = {
            'SBrkr': 'Standard Circuit Breakers & Romex',
            'FuseA': 'Fuse Box over 60 AMP and all Romex wiring (Average)',
            'FuseF': '60 AMP Fuse Box and mostly Romex wiring (Fair)',
            'FuseP': '60 AMP Fuse Box and mostly knob & tube wiring (poor)',
            'Mix': 'Mixed'
        }

        kitchenqual_mapping = {
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Typical/Average',
            'Fa': 'Fair',
            'Po': 'Poor'
        }

        functional_mapping = {
            'Typ': 'Typical Functionality',
            'Min1': 'Minor Deductions 1',
            'Min2': 'Minor Deductions 2',
            'Mod': 'Moderate Deductions',
            'Maj1': 'Major Deductions 1',
            'Maj2': 'Major Deductions 2',
            'Sev': 'Severely Damaged',
            'Sal': 'Salvage only'
        }

        fireplacequ_mapping = {
            'Ex': 'Excellent - Exceptional Masonry Fireplace',
            'Gd': 'Good - Masonry Fireplace in main level',
            'TA': 'Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement',
            'Fa': 'Fair - Prefabricated Fireplace in basement',
            'Po': 'Poor - Ben Franklin Stove',
            'NA': 'No Fireplace'
        }

        garagetype_mapping = {
            '2Types': 'More than one type of garage',
            'Attchd': 'Attached to home',
            'Basment': 'Basement Garage',
            'BuiltIn': 'Built-In (Garage part of house - typically has room above garage)',
            'CarPort': 'Car Port',
            'Detchd': 'Detached from home',
            'NA': 'No Garage'
        }

        garagefinish_mapping = {
            'Fin': 'Finished',
            'RFn': 'Rough Finished',
            'Unf': 'Unfinished',
            'NA': 'No Garage'
        }

        garagequal_mapping = {
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Typical/Average',
            'Fa': 'Fair',
            'Po': 'Poor',
            'NA': 'No Garage'
        }

        garagecond_mapping = {
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Typical/Average',
            'Fa': 'Fair',
            'Po': 'Poor',
            'NA': 'No Garage'
        }

        paveddrive_mapping = {
            'Y': 'Paved',
            'P': 'Partial Pavement',
            'N': 'Dirt/Gravel'
        }

        poolqc_mapping = {
            'Ex': 'Excellent',
            'Gd': 'Good',
            'TA': 'Average/Typical',
            'Fa': 'Fair',
            'NA': 'No Pool'
        }

        fence_mapping = {
            'GdPrv': 'Good Privacy',
            'MnPrv': 'Minimum Privacy',
            'GdWo': 'Good Wood',
            'MnWw': 'Minimum Wood/Wire',
            'NA': 'No Fence'
        }

        miscfeature_mapping = {
            'Elev': 'Elevator',
            'Gar2': '2nd Garage (if not described in garage section)',
            'Othr': 'Other',
            'Shed': 'Shed (over 100 SF)',
            'TenC': 'Tennis Court',
            'NA': 'None'
        }

        saletype_mapping = {
            'WD': 'Warranty Deed - Conventional',
            'CWD': 'Warranty Deed - Cash',
            'VWD': 'Warranty Deed - VA Loan',
            'New': 'Home just constructed and sold',
            'COD': 'Court Officer Deed/Estate',
            'Con': 'Contract 15% Down payment regular terms',
            'ConLw': 'Contract Low Down payment and low interest',
            'ConLI': 'Contract Low Interest',
            'ConLD': 'Contract Low Down',
            'Oth': 'Other'
        }

        salecondition_mapping = {
            'Normal': 'Normal Sale',
            'Abnorml': 'Abnormal Sale - trade, foreclosure, short sale',
            'AdjLand': 'Adjoining Land Purchase',
            'Alloca': 'Allocation - two linked properties with separate deeds, typically condo with a garage unit',
            'Family': 'Sale between family members',
            'Partial': 'Home was not completed when last assessed (associated with New Homes)'
        }


        col_dict = {
            "MSSubClass": "tType of dwelling involved in the sale.",
            "MSZoning": "General zoning classification of the sale.",
            "LotFrontage": "Linear feet of street connected to property",
            "LotArea": "Lot size in square feet",
            "Street": "Type of road access to property",
            "Alley": "Type of alley access to property",
            "LotShape": "General shape of property",
            "LandContour": "Flatness of the property",
            "Utilities": "Type of utilities available",
            "LotConfig": "Lot configuration",
            "LandSlope": "Slope of property",
            "Neighborhood": "Physical locations within Ames city limits",
            "Condition1": "Proximity to various conditions",
            "Condition2": "Proximity to various conditions (if more than one is present)",
            "BldgType": "Type of dwelling",
            "HouseStyle": "Style of dwelling",
            "OverallQual": "Overall material and finish of the house",
            "OverallCond": "Overall condition of the house",
            "YearBuilt": "Original construction date",
            "YearRemodAdd": "Remodel date (same as construction date if no remodeling or additions)",
            "RoofStyle": "Type of roof",
            "RoofMatl": "Roof material",
            "Exterior1st": "Exterior covering on house",
            "Exterior2nd": "Exterior covering on house (if more than one material)",
            "MasVnrType": "Masonry veneer type",
            "MasVnrArea": "Masonry veneer area in square feet",
            "ExterQual": "Quality of the material on the exterior",
            "ExterCond": "Present condition of the material on the exterior",
            "Foundation": "Type of foundation",
            "BsmtQual": "Height of the basement",
            "BsmtCond": "General condition of the basement",
            "BsmtExposure": "Walkout or garden level walls",
            "BsmtFinType1": "Basement finished area",
            "BsmtFinSF1": "Type 1 finished square feet",
            "BsmtFinType2": "Basement finished area (if multiple types)",
            "BsmtFinSF2": "Type 2 finished square feet",
            "BsmtUnfSF": "Unfinished square feet of basement area",
            "TotalBsmtSF": "Total square feet of basement area",
            "Heating": "Type of heating",
            "HeatingQC": "Heating quality and condition",
            "CentralAir": "Central air conditioning",
            "Electrical": "Electrical system",
            "1stFlrSF": "First Floor square feet",
            "2ndFlrSF": "Second floor square feet",
            "LowQualFinSF": "Low quality finished square feet (all floors)",
            "GrLivArea": "Above grade (ground) living area square feet",
            "BsmtFullBath": "Basement full bathrooms",
            "BsmtHalfBath": "Basement half bathrooms",
            "FullBath": "Full bathrooms above grade",
            "HalfBath": "Half baths above grade",
            "Bedroom": "Bedrooms above grade (does NOT include basement bedrooms)",
            "Kitchen": "Kitchens above grade",
            "KitchenQual": "Kitchen quality",
            "TotRmsAbvGrd": "Total rooms above grade (does not include bathrooms)",
            "Functional": "Home functionality (Assume typical unless deductions are warranted)",
            "Fireplaces": "Number of fireplaces",
            "FireplaceQu": "Fireplace quality",
            "GarageType": "Garage location",
            "GarageYrBlt": "Year garage was built",
            "GarageFinish": "Interior finish of the garage",
            "GarageCars": "Size of garage in car capacity",
            "GarageArea": "Size of garage in square feet",
            "GarageQual": "Garage quality",
            "GarageCond": "Garage condition",
            "PavedDrive": "Paved driveway",
            "WoodDeckSF": "Wood deck area in square feet",
            "OpenPorchSF": "Open porch area in square feet",
            "EnclosedPorch": "Enclosed porch area in square feet",
            "3SsnPorch": "Three season porch area in square feet",
            "ScreenPorch": "Screen porch area in square feet",
            "PoolArea": "Pool area in square feet",
            "PoolQC": "Pool quality",
            "Fence": "Fence quality",
            "MiscFeature": "Miscellaneous feature not covered in other categories",
            "MiscVal": "$Value of miscellaneous feature",
            "MoSold": "Month Sold (MM)",
            "YrSold": "Year Sold (YYYY)",
            "SaleType": "Type of sale",
            "SaleCondition": "Condition of sale"
        }

        for col in df.columns:
            variable_name = f'{col.lower()}_mapping'
            if variable_name in locals() or variable_name in globals():
                df.loc[:, col] = df[col].map(locals()[variable_name])

        df = df.rename(columns=col_dict)
        self.old_to_new_col_mapping = col_dict

        return df



if __name__ == '__main__':
    houseprices = HousePricesData()
    houseprices.setup()

    print(houseprices.n_features)