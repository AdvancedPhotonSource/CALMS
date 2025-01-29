import robotics as ro
import logging
import time
import loca

logging.basicConfig(level=logging.INFO)

def move_substrate_to_coating_station():
    """Pick up a substrate and move it to the coating station."""
    try:
        # Initialize the robot controller
        c9 = ro.system.init('controller')
        logging.info("Robot controller initialized")

        # Pick up the substrate tool
        c9.tool = 'substrate_tool'
        logging.info("Substrate tool picked up")

        # Move to the substrate rack
        c9.position = loca.substrate_rack_seq[0, 0]
        logging.info("Moved to substrate rack")

        # Activate vacuum to pick up substrate
        c9.set_output('substrate_tool', True)
        time.sleep(0.5)  # Wait for vacuum to stabilize

        # Move to the coating station
        c9.position = loca.s_coater
        logging.info("Moved to coating station")

        # Activate vacuum on coater stage
        c9.set_output('coater_stage_vacuum', True)
        logging.info("Coater stage vacuum activated")

        # Release substrate
        c9.set_output('substrate_tool', False)
        logging.info("Substrate released on coater stage")

        # Drop off the substrate tool
        c9.tool = None
        logging.info("Substrate tool dropped off")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    move_substrate_to_coating_station()
