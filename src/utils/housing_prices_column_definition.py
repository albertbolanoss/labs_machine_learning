import numpy as np

class ColumnInfo:
    def __init__(self, name, desc, dtype, format_str=None):
        """
        Initializes the feature information.
        
        Args:
            name (str): The name of the column.
            desc (str): The description of the feature.
            dtype (type): The expected data type (e.g., np.int64, str).
            format_str (str, optional): The format string for display.
        """
        self.name = name
        self.desc = desc
        self.dtype = dtype
        self.format_str = format_str

class Column:
    PRICE = ColumnInfo("Price", "The price of the house.", np.int64, format_str='$ {:,.4f}')
    AREA = ColumnInfo("Area", "The area of the property in square feet.", np.int64, format_str='{:,.0f}')
    LOCATION = ColumnInfo("Location", "The neighborhood in Hyderabad.", str)
    NO_OF_BEDROOMS = ColumnInfo("No. of Bedrooms", "The number of bedrooms.", np.int64, format_str='{:.0f}')
    RESALE = ColumnInfo("Resale", "A binary flag indicating if the property is for resale.", np.int64, format_str='{:.0f}')
    MAINTENANCE_STAFF = ColumnInfo("MaintenanceStaff", "A flag for the availability of maintenance staff.", np.int64, format_str='{:.0f}')
    GYMNASIUM = ColumnInfo("Gymnasium", "A flag for the availability of a gymnasium.", np.int64, format_str='{:.0f}')
    SWIMMING_POOL = ColumnInfo("SwimmingPool", "A flag for the availability of a swimming pool.", np.int64, format_str='{:.0f}')
    LANDSCAPED_GARDENS = ColumnInfo("LandscapedGardens", "A flag for the availability of landscaped gardens.", np.int64, format_str='{:.0f}')
    JOGGING_TRACK = ColumnInfo("JoggingTrack", "A flag for the availability of a jogging track.", np.int64, format_str='{:.0f}')
    RAIN_WATER_HARVESTING = ColumnInfo("RainWaterHarvesting", "A flag for the availability of rainwater harvesting.", np.int64, format_str='{:.0f}')
    INDOOR_GAMES = ColumnInfo("IndoorGames", "A flag for the availability of indoor games facilities.", np.int64, format_str='{:.0f}')
    SHOPPING_MALL = ColumnInfo("ShoppingMall", "A flag for the availability of a nearby shopping mall.", np.int64, format_str='{:.0f}')
    INTERCOM = ColumnInfo("Intercom", "A flag for the availability of an intercom facility.", np.int64, format_str='{:.0f}')
    SPORTS_FACILITY = ColumnInfo("SportsFacility", "A flag for the availability of a sports facility.", np.int64, format_str='{:.0f}')
    ATM = ColumnInfo("ATM", "A flag for the availability of a nearby ATM.", np.int64, format_str='{:.0f}')
    CLUB_HOUSE = ColumnInfo("ClubHouse", "A flag for the availability of a club house.", np.int64, format_str='{:.0f}')
    SCHOOL = ColumnInfo("School", "A flag for the availability of a nearby school.", np.int64, format_str='{:.0f}')
    SECURITY_24X7 = ColumnInfo("24X7Security", "A flag for the availability of 24x7 security.", np.int64, format_str='{:.0f}')
    POWER_BACKUP = ColumnInfo("PowerBackup", "A flag for the availability of power backup.", np.int64, format_str='{:.0f}')
    CAR_PARKING = ColumnInfo("CarParking", "A flag for the availability of car parking.", np.int64, format_str='{:.0f}')
    STAFF_QUARTER = ColumnInfo("StaffQuarter", "A flag for the availability of a staff quarter.", np.int64, format_str='{:.0f}')
    CAFETERIA = ColumnInfo("Cafeteria", "A flag for the availability of a cafeteria.", np.int64, format_str='{:.0f}')
    MULTIPURPOSE_ROOM = ColumnInfo("MultipurposeRoom", "A flag for the availability of a multipurpose room.", np.int64, format_str='{:.0f}')
    HOSPITAL = ColumnInfo("Hospital", "A flag for the availability of a nearby hospital.", np.int64, format_str='{:.0f}')
    WASHING_MACHINE = ColumnInfo("WashingMachine", "A flag indicating if a washing machine is included.", np.int64, format_str='{:.0f}')
    GAS_CONNECTION = ColumnInfo("Gasconnection", "A flag for the availability of a gas connection.", np.int64, format_str='{:.0f}')
    AC = ColumnInfo("AC", "A flag indicating if air conditioning is included.", np.int64, format_str='{:.0f}')
    WIFI = ColumnInfo("Wifi", "A flag for the availability of WiFi.", np.int64, format_str='{:.0f}')
    CHILDRENS_PLAY_AREA = ColumnInfo("Children'splayarea", "A flag for the availability of a children's play area.", np.int64, format_str='{:.0f}')
    LIFT_AVAILABLE = ColumnInfo("LiftAvailable", "A flag for the availability of a lift.", np.int64, format_str='{:.0f}')
    BED = ColumnInfo("BED", "A flag indicating if a bed is included.", np.int64, format_str='{:.0f}')
    VAASTU_COMPLIANT = ColumnInfo("VaastuCompliant", "A flag indicating if the property is Vaastu compliant.", np.int64, format_str='{:.0f}')
    MICROWAVE = ColumnInfo("Microwave", "A flag indicating if a microwave is included.", np.int64, format_str='{:.0f}')
    GOLF_COURSE = ColumnInfo("GolfCourse", "A flag for the availability of a golf course.", np.int64, format_str='{:.0f}')
    TV = ColumnInfo("TV", "A flag indicating if a TV is included.", np.int64, format_str='{:.0f}')
    DINING_TABLE = ColumnInfo("DiningTable", "A flag indicating if a dining table is included.", np.int64, format_str='{:.0f}')
    SOFA = ColumnInfo("Sofa", "A flag indicating if a sofa is included.", np.int64, format_str='{:.0f}')
    WARDROBE = ColumnInfo("Wardrobe", "A flag indicating if a wardrobe is included.", np.int64, format_str='{:.0f}')
    REFRIGERATOR = ColumnInfo("Refrigerator", "A flag indicating if a refrigerator is included.", np.int64, format_str='{:.0f}')

    @staticmethod
    def get_format_dict():
        """
        Generates a format dictionary from the features defined in the class
        that have a format_str defined.
        """
        return {
            value.name: value.format_str
            for value in Column.__dict__.values()
            if isinstance(value, ColumnInfo) and value.format_str is not None
        }

    @staticmethod
    def get_dtypes_dict():
        """
        Generates a data type dictionary for all features defined in the class.
        """
        return {
            value.name: value.dtype
            for value in Column.__dict__.values()
            if isinstance(value, ColumnInfo)
        }
    
    @staticmethod
    def show_expected_columns_details():
        """
        Show column informarmation as the colunm name, description, expected type, format
        """
        for attribute_name in dir(Column):
            attribute = getattr(Column, attribute_name)
            
            if isinstance(attribute, ColumnInfo):
                print(f"\nColumn: {attribute.name}")
                print(f"  - Description: {attribute.desc}")
                print(f"  - Expected Type: {attribute.dtype.__name__}")
                
                expected_format = attribute.format_str if attribute.format_str else "No defined"
                print(f"  - Expected Format: {expected_format}")