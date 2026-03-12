import mmap
import struct
import json


def convertDegreeArcToPercent(value):
    return max(value / 360, 0)


class AssettoCorsaData:
    def __init__(self):
        print('[Driver] Inizializzazione AssettoCorsaData...')
        self.fields = 'packetId throttle brake fuel gear rpm steerAngle speed velocity1 velocity2 velocity3 accGX accGY accGZ wheelSlipFL wheelSlipFR wheelSlipRL wheelSlipRR wheelLoadFL wheelLoadFR wheelLoadRL wheelLoadRR wheelsPressureFL wheelsPressureFR wheelsPressureRL wheelsPressureRR wheelAngularSpeedFL wheelAngularSpeedFR wheelAngularSpeedRL wheelAngularSpeedRR TyrewearFL TyrewearFR TyrewearRL TyrewearRR tyreDirtyLevelFL tyreDirtyLevelFR tyreDirtyLevelRL tyreDirtyLevelRR TyreCoreTempFL TyreCoreTempFR TyreCoreTempRL TyreCoreTempRR camberRADFL camberRADFR camberRADRL camberRADRR suspensionTravelFL suspensionTravelFR suspensionTravelRL suspensionTravelRR drs tc1 heading pitch roll cgHeight carDamagefront carDamagerear carDamageleft carDamageright carDamagecentre numberOfTyresOut pitLimiterOn abs1 kersCharge kersInput automat rideHeightfront rideHeightrear turboBoost ballast airDensity airTemp roadTemp localAngularVelX localAngularVelY localAngularVelZ finalFF performanceMeter engineBrake ersRecoveryLevel ersPowerLevel ersHeatCharging ersIsCharging kersCurrentKJ drsAvailable drsEnabled brakeTempFL brakeTempFR brakeTempRL brakeTempRR clutch tyreTempI1 tyreTempI2 tyreTempI3 tyreTempI4 tyreTempM1 tyreTempM2 tyreTempM3 tyreTempM4 tyreTempO1 tyreTempO2 tyreTempO3 tyreTempO4 isAIControlled tyreContactPointFLX tyreContactPointFLY tyreContactPointFLZ tyreContactPointFRX tyreContactPointFRY tyreContactPointFRZ tyreContactPointRLX tyreContactPointRLY tyreContactPointRLZ tyreContactPointRRX tyreContactPointRRY tyreContactPointRRZ tyreContactNormalFLX tyreContactNormalFLY tyreContactNormalFLZ tyreContactNormalFRX tyreContactNormalFRY tyreContactNormalFRZ tyreContactNormalRLX tyreContactNormalRLY tyreContactNormalRLZ tyreContactNormalRRX tyreContactNormalRRY tyreContactNormalRRZ tyreContactHeadingFLX tyreContactHeadingFLY tyreContactHeadingFLZ tyreContactHeadingFRX tyreContactHeadingFRY tyreContactHeadingFRZ tyreContactHeadingRLX tyreContactHeadingRLY tyreContactHeadingRLZ tyreContactHeadingRRX tyreContactHeadingRRY tyreContactHeadingRRZ brakeBias localVelocityX localVelocityY localVelocityZ P2PActivation P2PStatus currentMaxRpm mz1 mz2 mz3 mz4 fx1 fx2 fx3 fx4 fy1 fy2 fy3 fy4 slipRatio1 slipRatio2 slipRatio3 slipRatio4 slipAngle1 slipAngle2 slipAngle3 slipAngle4 tcinAction absInAction suspensionDamage1 suspensionDamage2 suspensionDamage3 suspensionDamage4 tyreTemp1 tyreTemp2 tyreTemp3 tyreTemp4 waterTemp brakePressureFL brakePressureFR brakePressureRL brakePressureRR frontBrakeCompound rearBrakeCompound padLifeFL padLifeFR padLifeRL padLifeRR discLifeFL discLifeFR discLifeRL discLifeRR'.split()
        self.layout = 'ifffiiffffffff 4f fffffffffffffffffffffffffffffffffffffffffffiifffiffffffffffffiiiiifiifffffffffffffffffiffffffffffffffffffffffffffffffffffffffffiifffffffffffffffffffffiifffffffffffffiiffffffff'
        self.physics_shm_size = struct.calcsize(self.layout)
        self.mmapPhysic = None

        # Pre-inizializza gli attributi a 0
        for field in self.fields:
            setattr(self, field, 0.0)

    def start(self):
        print('[Driver] Connessione in corso...')
        if not self.mmapPhysic:
            self.mmapPhysic = mmap.mmap(-1, self.physics_shm_size, "Local\\acpmf_physics", access=mmap.ACCESS_READ)

    def update(self):
        """Legge la memoria condivisa e aggiorna gli attributi della classe. Permette la sintassi asm.rpm"""
        if not self.mmapPhysic:
            return

        self.mmapPhysic.seek(0)
        rawData = self.mmapPhysic.read(self.physics_shm_size)
        unpacked = struct.unpack(self.layout, rawData)

        # Mappa i dati estratti sui nomi dei campi
        data_dict = {self.fields[i]: val for i, val in enumerate(unpacked)}

        # Converte i gruppi in array (es. brakeTempFL, FR, RL, RR -> brakeTemp = [FL, FR, RL, RR])
        for newName in ['wheelSlip', 'wheelLoad', 'wheelsPressure',
                        'brakeTemp', 'brakePressure', 'Tyrewear', 'wheelAngularSpeed',
                        'padLife', 'discLife', 'camberRAD', 'TyreCoreTemp', 'tyreDirtyLevel',
                        'suspensionTravel']:

            data_dict[newName] = []
            for oldName in [newName + 'FL', newName + 'FR', newName + 'RL', newName + 'RR']:
                if oldName in data_dict:
                    data_dict[newName].append(convertDegreeArcToPercent(data_dict[oldName]))
                    del data_dict[oldName]

        # Imposta dinamicamente gli attributi (questo ti permette di usare self.asm.rpm, self.asm.speed, ecc.)
        for key, value in data_dict.items():
            setattr(self, key, value)

    def stop(self):
        print('[Driver] Chiusura connessione...')
        if self.mmapPhysic:
            self.mmapPhysic.close()
        self.mmapPhysic = None


# Test veloce per assicurarsi che funzioni
if __name__ == '__main__':
    import time

    reader = AssettoCorsaData()
    reader.start()
    while True:
        reader.update()  # Aggiorna i dati
        print(f"RPM: {reader.rpm:.0f} | Speed: {reader.speed:.1f} km/h | Gear: {reader.gear}")
        time.sleep(0.1)