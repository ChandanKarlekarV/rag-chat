"""Generate the sample PDFs used for demos and evaluation.

* `samples/halcyon_h2_handbook.pdf` - a fictional home-battery manual (the evaluation target).
* `samples/solvane_w7_charger_guide.pdf` - a fictional EV-charger guide that deliberately covers
  the same topics (Wi-Fi, warranty, status lights, error codes, installation) as a hard distractor.

All text is original and written for this project (no third-party copyright). Every page carries a
running header and footer so the cleaner's boilerplate removal is exercised.

Usage: python scripts/make_sample_pdf.py [output_dir]
"""

import sys
from pathlib import Path

import pymupdf

TITLE = "Halcyon H2 Home Battery - Owner's Handbook"

PAGES: list[tuple[str, str]] = [
    (
        "1. Welcome and Product Overview",
        """Thank you for choosing the Halcyon H2, a wall-mounted home battery designed to store solar energy during the day and supply it to your home in the evening or during a power cut.

The H2 uses lithium iron phosphate (LFP) cells, a chemistry chosen for its thermal stability and long service life. Each unit provides 11.2 kWh of usable energy capacity. It can deliver 5.8 kW of continuous power and up to 7.6 kW of peak power for ten seconds, which is enough to start most household motors such as well pumps and air conditioners.

The battery enclosure weighs 118 kg and measures 1150 mm high, 755 mm wide and 160 mm deep. It is rated IP56, so it may be installed indoors or outdoors when sheltered from direct rain.

Up to four H2 units can be connected together on a single site, giving a maximum stored energy of 44.8 kWh. Additional units are added in parallel and are managed automatically by the Halcyon Link gateway.

This handbook explains how the H2 works, how to operate it safely, and what to do if something goes wrong. Installation must be carried out by a certified installer; the installation chapter is provided for reference only.""",
    ),
    (
        "2. Safety Information",
        """Read all safety instructions before operating the H2. Failure to follow these instructions may result in fire, electric shock or serious injury.

The H2 is designed to operate at ambient temperatures between -10 degrees C and 50 degrees C. Outside this range the battery management system reduces charging and discharging power to protect the cells. Charging is completely disabled below -5 degrees C until the internal cell heater has warmed the pack.

Do not install the unit within 90 cm of a gas meter, gas regulator or any other source of flammable vapour. Keep at least 30 cm of free space above the enclosure and 15 cm on each side for ventilation.

Never open the enclosure. There are no user-serviceable parts inside, and the internal DC bus can hold a dangerous voltage even when the unit appears to be off.

In an emergency, press the red Emergency Stop button on the right-hand side of the enclosure. This isolates the battery cells from the inverter within one second. Then contact your installer before turning the unit back on.

Children and pets should be kept away from the unit. Do not place objects on top of the enclosure or lean anything against the cooling vents.""",
    ),
    (
        "3. Installation Requirements",
        """The H2 must be mounted on a wall made of concrete, solid brick or masonry block that can carry at least four times the weight of the unit. Timber-frame or plasterboard walls are not suitable without an approved floor stand.

The wall bracket, part number HB-WM2, is supplied in the box together with four M10 expansion anchors. The bottom of the enclosure should sit 60 cm above the finished floor in areas at risk of flooding, and at least 10 cm above the floor in all other locations.

Only installers who hold a valid Halcyon Certified Installer card may commission the system. Commissioning registers the product serial number and activates the warranty. Systems installed by uncertified persons are not covered by the warranty.

The installer will connect the H2 to the main switchboard through a dedicated 40 A circuit breaker. An optional Backup Switch, part number HB-BS1, separates essential household circuits so that they remain powered during a grid outage.

Allow approximately four hours for a standard single-unit installation, including commissioning and a full functional test.""",
    ),
    (
        "4. Connectivity and the Halcyon Link Gateway",
        """Every H2 system includes a Halcyon Link gateway, a small box that connects the battery to your home network and to the Halcyon cloud service. The gateway communicates with the battery over a shielded CAN bus cable.

The gateway supports Wi-Fi on the 2.4 GHz band only; 5 GHz networks are not supported. For the most reliable connection, use the wired Ethernet port, which takes priority automatically when a cable is plugged in. A built-in LTE cellular modem provides a backup connection if both Wi-Fi and Ethernet are unavailable.

The Link gateway stores up to 30 days of energy data locally, so no history is lost during an internet outage. Data is uploaded to the cloud as soon as the connection is restored.

To pair the gateway, hold the Link button for five seconds until the status light flashes blue, then follow the instructions in the Halcyon app. Pairing mode times out after ten minutes.

The gateway also exposes a local Modbus TCP interface on port 502 for integration with third-party home energy management systems. This interface is read-only by default and must be enabled by your installer.""",
    ),
    (
        "5. Operating Modes",
        """The H2 offers three operating modes, which you can select at any time in the Halcyon app.

Self-Powered mode stores excess solar energy and uses it to power your home when the sun goes down. This mode maximises your use of your own solar generation and is recommended for most households.

Time-Based Control mode uses your electricity tariff to decide when to charge and discharge. The battery charges when electricity is cheap and powers your home when prices are high. You enter your tariff periods in the app, or select your retailer from the list.

Backup Only mode keeps the battery fully charged at all times and only discharges during a grid outage. Choose this mode if protection against power cuts is your main priority.

In Self-Powered and Time-Based Control modes, a portion of energy is always held in reserve for outages. The default backup reserve is 20 percent of capacity. You can change the reserve in the app to any value between 0 and 100 percent in steps of 5 percent.

Mode changes take effect within one minute. The app shows a live energy flow diagram so you can see where your power is coming from at any moment.""",
    ),
    (
        "6. Backup Power During Outages",
        """When the grid fails, the H2 disconnects from the grid and begins powering your backed-up circuits automatically. The switch-over completes in less than 150 milliseconds, so most appliances, computers and lights continue running without interruption.

During an outage the H2 can still recharge from your solar panels. If the battery becomes completely empty and the sun is not shining, the system shuts down and restarts automatically once solar power returns or the grid is restored.

Storm Watch is an optional feature that monitors severe weather warnings for your area. When a warning is issued, Storm Watch charges the battery to 100 percent ahead of the storm, regardless of the selected operating mode. You will receive a notification in the app when Storm Watch is active. Storm Watch can be turned off in the app settings.

To make your stored energy last longer during an outage, avoid running high-power appliances such as ovens, electric water heaters and clothes dryers. Large loads above the 5.8 kW continuous rating will cause the system to shut down temporarily to protect itself.

When grid power returns, the H2 waits five minutes to confirm the grid is stable before reconnecting.""",
    ),
    (
        "7. Routine Maintenance",
        """The H2 requires very little maintenance. The battery management system continuously monitors the health of every cell and balances them automatically.

Firmware updates are delivered over the internet and installed automatically. Updates are checked once a month and are installed at 2:00 AM local time to avoid disrupting your day. During an update, which takes around fifteen minutes, the battery continues to supply backup power but does not charge or discharge for normal use.

Every six months, inspect the cooling vents on the sides of the enclosure and gently remove dust, leaves or cobwebs with a soft brush. Do not use water, solvents or a pressure washer on the unit.

The outside of the enclosure may be wiped with a damp cloth. Make sure the vents are not blocked by plants, stored items or snow.

We recommend a professional inspection by a certified installer every five years. The inspection includes a torque check of all electrical terminals and a capacity test.""",
    ),
    (
        "8. Status Lights and Troubleshooting",
        """The status light on the front of the H2 shows the current state of the system.

A solid green light means the system is operating normally. A slowly pulsing green light means the battery is charging. A blinking amber light indicates a grid fault: the grid voltage or frequency is outside the allowed range and the H2 is running in backup mode. A solid red light means thermal shutdown: the unit has become too hot and has stopped operating until it cools down. Check that the vents are clear and that the unit is not in direct afternoon sun.

The app may also show an error code. Error E-214 indicates an insulation (isolation) fault on the solar or battery wiring; the system stops to protect you and an installer must inspect the wiring. Error E-101 indicates a lost connection between the battery and the Link gateway; check the CAN bus cable and restart the gateway. Error E-330 means the backup reserve setting could not be applied; reopen the app and set the reserve again.

To restart the system, switch off the H2 using the side switch, wait thirty seconds, then switch it on again. If the problem continues, contact Halcyon support through the app.""",
    ),
    (
        "9. Warranty",
        """Halcyon warrants the H2 against defects in materials and workmanship for 12 years from the date of installation.

Halcyon also guarantees that the battery will retain at least 70 percent of its original usable capacity by the end of the warranty period, provided the total energy delivered by the battery does not exceed 37.8 MWh. The warranty ends when either the 12-year period expires or the throughput limit is reached, whichever comes first.

The warranty does not cover damage caused by flooding, fire, lightning, improper installation by an uncertified person, opening the enclosure, or operation outside the specified temperature range.

To make a warranty claim, contact Halcyon support with your product serial number, which is printed on the label under the enclosure and is also shown in the app. Halcyon will diagnose the fault remotely where possible and will arrange repair or replacement at no cost if the claim is approved.

The warranty is transferable to a new owner if the property is sold. The new owner should register the transfer in the app within 90 days.""",
    ),
    (
        "10. Technical Specifications and Recycling",
        """Cell chemistry: lithium iron phosphate (LFP). Usable capacity: 11.2 kWh. Continuous power: 5.8 kW. Peak power: 7.6 kW for 10 seconds. Round-trip efficiency: 90 percent. Nominal battery voltage: 409 V. Design cycle life: 8,000 cycles.

Operating temperature: -10 to 50 degrees C. Enclosure rating: IP56. Noise level: under 30 dB at one metre. Weight: 118 kg. Dimensions: 1150 x 755 x 160 mm. Communication: CAN bus to gateway; Wi-Fi 2.4 GHz, Ethernet and LTE at the gateway.

At the end of its life, the H2 must not be disposed of with household waste. Halcyon operates a free take-back programme: contact support to schedule a collection. Collected batteries are sent to a specialist facility where more than 95 percent of the cell materials are recovered for reuse.

The H2 complies with IEC 62619 for the safety of industrial lithium batteries and with UL 9540 for energy storage systems.""",
    ),
]

DISTRACTOR_TITLE = "Solvane W7 Wallbox EV Charger - User Guide"

DISTRACTOR_PAGES: list[tuple[str, str]] = [
    (
        "1. About Your W7 Wallbox",
        """The Solvane W7 is a home charging station for electric vehicles. It delivers up to 7.4 kW on a single-phase supply or 22 kW on a three-phase supply, depending on how it is installed.

The W7 has a 7.5 metre tethered Type 2 cable, so there is no need to carry a separate cable for home charging. The enclosure is rated IP65 and IK10 and can be mounted outdoors on a garage wall, a fence post or a dedicated pedestal.

The charger weighs 6.8 kg including the cable. A built-in energy meter records every session, and you can review your charging history and costs in the Solvane app.

The W7 can work alongside a home battery or solar system. In Solar Boost mode it charges your car only from surplus solar energy that would otherwise be exported to the grid.""",
    ),
    (
        "2. Installation",
        """The W7 must be installed by a qualified electrician in accordance with local wiring rules. A dedicated circuit protected by a 32 A Type A RCBO is required for single-phase installations.

Mount the charger so that the cable holster sits between 80 cm and 120 cm above the ground. Use the supplied wall plate and the four 6 mm screws; masonry, concrete and timber surfaces are all suitable when the correct fixings are used.

A load management clamp, part number SV-CT1, can be fitted around the main supply cable. It lets the W7 reduce its charging current automatically so that your home never exceeds the capacity of the main fuse.

A typical installation takes two to three hours. After installation the electrician activates the charger in the Solvane Pro app and records the maximum current setting.""",
    ),
    (
        "3. Connecting to Wi-Fi and the App",
        """The W7 connects to your home network over Wi-Fi on either the 2.4 GHz or the 5 GHz band. A wired Ethernet port is available under the front cover, and an optional 4G module can be fitted for locations without Wi-Fi coverage.

To connect, open the Solvane app, choose Add Charger and scan the QR code printed inside the front cover. The charger creates a temporary hotspot during setup, which closes automatically after fifteen minutes.

Once connected, the app lets you start and stop charging, set a charging schedule, lock the charger with a PIN, and view live power. Up to five household members can share access to the same charger.

The W7 supports the OCPP 1.6J protocol, so it can also be managed by third-party charging platforms if you use the charger for business purposes.""",
    ),
    (
        "4. Everyday Charging",
        """To charge, park your vehicle, take the connector from the holster and plug it into your car's charging inlet. Charging starts automatically unless a schedule or PIN lock is active.

Scheduled charging lets you choose start and finish times so that charging happens when electricity is cheapest, for example overnight. If you set a departure time instead, the W7 calculates when to start so that the car is ready in time.

You can stop charging at any time from the app or from your car. When charging is finished, unplug the connector from the car first and return it to the holster to protect the pins from dirt and moisture.

Never use an extension lead or an adapter with the W7. Do not charge if the cable or connector is visibly damaged.""",
    ),
    (
        "5. Indicator Light and Error Codes",
        """A light ring around the front of the W7 shows its status. A solid white ring means the charger is ready. A pulsing blue ring means the vehicle is charging, and a solid blue ring means charging is complete or paused by the car.

A solid red ring indicates a ground fault: the charger has detected leakage current and has stopped for safety. Unplug the vehicle, wait one minute and try again; if the fault returns, contact your electrician. A flashing yellow ring means the charger is waiting for a scheduled start time.

Error codes are shown in the app. Error E-201 means the supply voltage is too low. Error E-214 means the connector temperature sensor has reported overheating; check that the connector pins are clean. Error E-305 indicates a communication problem with the vehicle; unplug the connector and plug it in again.

To restart the W7, switch off its circuit breaker for thirty seconds and then switch it on again.""",
    ),
    (
        "6. Care and Maintenance",
        """Keep the connector clean and dry. Wipe the charger and cable with a soft damp cloth when needed; do not use abrasive cleaners, solvents or high-pressure water.

Check the cable for cuts or kinks once a month. A damaged cable must be replaced by an electrician using a genuine Solvane replacement cable.

Firmware updates are downloaded automatically over Wi-Fi and installed when no vehicle is connected. You can also check for updates manually in the app under Charger Settings.

An annual test of the built-in residual current protection is recommended. Press the TEST button under the front cover; the light ring turns red and charging stops. Press RESET to restore normal operation.""",
    ),
    (
        "7. Warranty and Support",
        """The Solvane W7 is covered by a three-year limited warranty from the date of purchase. The warranty can be extended to five years by registering the charger in the app within 30 days of installation.

The warranty covers defects in materials and workmanship. It does not cover damage caused by vehicle collisions, misuse, incorrect installation or the use of adapters.

For support, use the Help section of the app or email support. Please have your charger serial number ready; it is printed on the label on the underside of the charger.

At the end of its life, return the W7 to a recycling centre for electrical equipment. Do not dispose of it with household waste.""",
    ),
    (
        "8. Specifications",
        """Rated power: 7.4 kW single-phase, 22 kW three-phase. Rated current: 32 A. Cable: 7.5 m tethered Type 2. Ingress protection: IP65. Impact protection: IK10.

Connectivity: Wi-Fi 2.4 GHz and 5 GHz, Ethernet, optional 4G. Protocols: OCPP 1.6J. Built-in protection: 6 mA DC leakage detection.

Operating temperature: -30 to 50 degrees C. Weight: 6.8 kg including cable. Dimensions: 300 x 200 x 110 mm.""",
    ),
]


def build(output: Path, title: str = TITLE, pages: list[tuple[str, str]] = PAGES, footer: str = "Rev. C") -> Path:
    """Render a manual to `output` and return the path."""
    doc = pymupdf.open()
    total = len(pages)
    for n, (heading, body) in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((50, 40), title, fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
        page.insert_text((50, 95), heading, fontsize=16, fontname="hebo")
        overflow = page.insert_textbox(
            pymupdf.Rect(50, 115, 545, 790), body, fontsize=10.5, fontname="helv", lineheight=1.35
        )
        if overflow < 0:
            raise RuntimeError(f"page {n} text does not fit")
        page.insert_text(
            (50, 815), f"Page {n} of {total} - {footer}", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    return output


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples")
    print(build(out_dir / "halcyon_h2_handbook.pdf"))
    print(build(out_dir / "solvane_w7_charger_guide.pdf", DISTRACTOR_TITLE, DISTRACTOR_PAGES, "Solvane Energy"))
