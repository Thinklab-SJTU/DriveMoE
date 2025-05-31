def get_scenario_id_13_experts(scenario_name):
    """Maps a scenario name to an expert ID for action Mixture-of-Experts (MoE) supervision.
    
    Args:
        scenario_name (str): Name of the driving scenario (e.g., "PedestrianCrossing").
    
    Returns:
        int: Expert ID (0-12) indicating which specialized policy should supervise 
             the action selection during training. Returns None if scenario_name is not found.
             
    Note:
        Expert IDs correspond to specialized sub-policies in the MoE framework:
        0: Merging expert
        1: ParkingExit expert
        2: Overtaking expert
        3: Emergency Brake expert
        4: Giveway expert
        5: Traffic Sign expert
        6-12: Scenario-specific experts
    """
    scenario_dict = {
        "CrossingBicycleFlow": 0,
        "EnterActorFlow": 0,
        "HighwayExit": 0,
        "InterurbanActorFlow": 0,
        "HighwayCutIn": 0,
        "InterurbanAdvancedActorFlow": 0,
        "MergerIntoSlowTrafficV2": 0,
        "MergerIntoSlowTraffic": 0, 
        "NonSignalizedJunctionLeftTurn": 11,
        "NonSignalizedJunctionRightTurn": 11, 
        "NonSignalizedJunctionLeftTurnEnterFlow": 0, 
        "LaneChange": 0,
        "SignalizedJunctionLeftTurn": 6, 
        "SignalizedJunctionRightTurn": 7,
        "SignalizedJunctionLeftTurnEnterFlow": 0,
        "ParkingExit": 1, 
        "Accident": 2,
        "AccidentTwoWays": 2, 
        "ConstructionObstacle": 2, 
        "ConstructionObstacleTwoWays": 2, 
        "HazardAtSideLaneTwoWays": 2, 
        "HazardAtSideLane": 2, 
        "ParkedObstacleTwoWays": 2, 
        "ParkedObstacle": 2, 
        "VehicleOpensDoorTwoWays": 2,
        "BlockedIntersection": 3,
        "DynamicObjectCrossing": 3, 
        "HardBreakRoute": 3, 
        "OppositeVehicleTakingPriority": 10, 
        "OppositeVehicleRunningRedLight": 12, 
        "ParkingCutIn": 3, 
        "PedestrianCrossing": 3,
        "ParkingCrossingPedestrian": 3, 
        "StaticCutIn": 3, 
        "VehicleTurningRoute": 8,
        "VehicleTurningRoutePedestrian": 9, 
        "ControlLoss": 3,
        "InvadingTurn": 4,
        "YieldToEmergencyVehicle": 4,
        "TJunction": 5,
        "VanillaSignalizedTurnEncounterGreenLight": 5,
        "VanillaSignalizedTurnEncounterRedLight": 5,
        "VanillaNonSignalizedTurn": 5,
        "VanillaNonSignalizedTurnEncounterStopsign": 5,
    }
    return scenario_dict.get(scenario_name)

def get_scenario_id_44_experts(scenario_name):
    scenario_dict = {
        "CrossingBicycleFlow": 0, # Mergeing expert
        "EnterActorFlow": 1,
        "HighwayExit": 2,
        "InterurbanActorFlow": 3,
        "HighwayCutIn": 4,
        "InterurbanAdvancedActorFlow": 5,
        "MergerIntoSlowTrafficV2": 6,
        "MergerIntoSlowTraffic": 7, 
        "NonSignalizedJunctionLeftTurn": 8,
        "NonSignalizedJunctionRightTurn": 9, 
        "NonSignalizedJunctionLeftTurnEnterFlow": 10, 
        "LaneChange": 11,
        "SignalizedJunctionLeftTurn": 12, 
        "SignalizedJunctionRightTurn": 13,
        "SignalizedJunctionLeftTurnEnterFlow": 14,
        "ParkingExit": 15, # ParkingExit expert
        "Accident": 16, # Overtaking expert
        "AccidentTwoWays": 17, 
        "ConstructionObstacle": 18, 
        "ConstructionObstacleTwoWays": 19, 
        "HazardAtSideLaneTwoWays": 20, 
        "HazardAtSideLane": 21, 
        "ParkedObstacleTwoWays": 22, 
        "ParkedObstacle": 23, 
        "VehicleOpensDoorTwoWays": 24,
        "BlockedIntersection": 25, # Emergency Brake expert
        "DynamicObjectCrossing": 26, 
        "HardBreakRoute": 27, 
        "OppositeVehicleTakingPriority": 28, 
        "OppositeVehicleRunningRedLight": 29, 
        "ParkingCutIn": 30, 
        "PedestrianCrossing": 31,
        "ParkingCrossingPedestrian": 32, 
        "StaticCutIn": 33, 
        "VehicleTurningRoute": 34,
        "VehicleTurningRoutePedestrian": 35, 
        "ControlLoss": 36,
        "InvadingTurn": 37, # Giveway expert
        "YieldToEmergencyVehicle": 38,
        "TJunction": 39, # Trafic Sign expert
        "VanillaSignalizedTurnEncounterGreenLight": 40,
        "VanillaSignalizedTurnEncounterRedLight": 41,
        "VanillaNonSignalizedTurn": 42,
        "VanillaNonSignalizedTurnEncounterStopsign": 43,
    }
    return scenario_dict.get(scenario_name)


def get_scenario_id_6_experts(scenario_name):
    scenario_dict = {
        "CrossingBicycleFlow": 0, # Mergeing expert
        "EnterActorFlow": 0,
        "HighwayExit": 0,
        "InterurbanActorFlow": 0,
        "HighwayCutIn": 0,
        "InterurbanAdvancedActorFlow": 0,
        "MergerIntoSlowTrafficV2": 0,
        "MergerIntoSlowTraffic": 0, 
        "NonSignalizedJunctionLeftTurn": 0,
        "NonSignalizedJunctionRightTurn": 0, 
        "NonSignalizedJunctionLeftTurnEnterFlow": 0, 
        "LaneChange": 0,
        "SignalizedJunctionLeftTurn": 0, 
        "SignalizedJunctionRightTurn": 0,
        "SignalizedJunctionLeftTurnEnterFlow": 0,
        "ParkingExit": 1, # ParkingExit expert
        "Accident": 2, # Overtaking expert
        "AccidentTwoWays": 2, 
        "ConstructionObstacle": 2, 
        "ConstructionObstacleTwoWays": 2, 
        "HazardAtSideLaneTwoWays": 2, 
        "HazardAtSideLane": 2, 
        "ParkedObstacleTwoWays": 2, 
        "ParkedObstacle": 2, 
        "VehicleOpensDoorTwoWays": 2,
        "BlockedIntersection": 3, # Emergency Brake expert
        "DynamicObjectCrossing": 3, 
        "HardBreakRoute": 3, 
        "OppositeVehicleTakingPriority": 3, 
        "OppositeVehicleRunningRedLight": 3, 
        "ParkingCutIn": 3, 
        "PedestrianCrossing": 3,
        "ParkingCrossingPedestrian": 3, 
        "StaticCutIn": 3, 
        "VehicleTurningRoute": 3,
        "VehicleTurningRoutePedestrian": 3, 
        "ControlLoss": 3,
        "InvadingTurn": 4, # Giveway expert
        "YieldToEmergencyVehicle": 4,
        "TJunction": 5, # Trafic Sign expert
        "VanillaSignalizedTurnEncounterGreenLight": 5,
        "VanillaSignalizedTurnEncounterRedLight": 5,
        "VanillaNonSignalizedTurn": 5,
        "VanillaNonSignalizedTurnEncounterStopsign": 5,
    }
    return scenario_dict.get(scenario_name)

def get_scenario_id(num_of_experts, scenario_name):
    func_dict = {
        6: get_scenario_id_6_experts,
        13: get_scenario_id_13_experts,
        44: get_scenario_id_44_experts,
    }
    if num_of_experts not in func_dict.keys():
        print(f"Invalid number of experts: '{num_of_experts}'. Available options: {list(func_dict.keys())}")
        return None
    return func_dict[num_of_experts](scenario_name)


def get_cam_id_by_name(cam_name):
    cam_dict = {
        "frontleft": 0,
        "frontright": 1,
        "back": 2,
        "backleft": 3,
        "backright": 4
    }
    return cam_dict.get(cam_name)

def get_cam_id_by_label(cam_name):
    cam_dict_2 = {
        "NULL": 2,
        "CAM_FRONT_LEFT": 0,
        "CAM_FRONT_RIGHT": 1,
        "CAM_BACK": 2,
        "CAM_BACK_LEFT": 3,
        "CAM_BACK_RIGHT": 4, 
    }
    return cam_dict_2.get(cam_name)