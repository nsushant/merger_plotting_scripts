import pynbody 
import darklight 
import pandas as pd 
import numpy as np 
import tangos 
from tangos.examples.mergers import *
from particle_tagging.edge.utils import *
import sys 
import os
from scipy.spatial.distance import cdist
### Function defs 
#pynbody.config["halo-class-priority"] = [pynbody.halo.ahf.AHFCatalogue]

def mutual(halo, IDs):
    tf = np.in1d(halo.d["iord"], IDs)
    return (len(tf[tf]) ** 2) / ((len(halo.d) * len(IDs)))


def findAHFhalonumACC(HOPhalonum,NameOfSim,Snap):
    
    dfhalonums = pd.read_csv(os.path.join("AHF_halonums/DMO",NameOfSim+"_accreted.csv")) 
    print(HOPhalonum,Snap)
    dfSnap = dfhalonums[dfhalonums["snapshot"] == Snap]
    print(dfSnap)
    
    AHFHaloNum = dfSnap[dfSnap["HOP halonum"] == float(HOPhalonum)]["AHF halonum"].values[0]
    print(dfSnap[dfSnap["HOP halonum"] == HOPhalonum])

    return AHFHaloNum
    

def findAHFhalonumINSITU(NameOfSim,Snap):

    dfhalonums = pd.read_csv(os.path.join("AHF_halonums/DMO",NameOfSim+".csv"))

    dfSnap = dfhalonums[dfhalonums["snapshot"] == Snap]

    AHFHaloNum = dfSnap["AHF halonum"].values[0]

    return AHFHaloNum
    
def group_mergers(z_merges,h_merges,q_merges):
    #groups the halo objects of merging halos by redshift                                                                                                   
    # quantities the function outputs 
    merging_halos_grouped_by_z = []
    qvalues_grouped_by_z = []
    m200_groupd_by_z = []
    z_unique_values = sorted(list(set(z_merges)))
    
    #internal array used to avoid halo overlaps
    list_of_selected_tuples = []

    # for each of these redshifts                                                                                                                
    for i in z_unique_values:
        # indices of halo objects merging at given redshift 'i'                                                                                              
        # zmerges = 1D (can contain 'i' multiple times)                                                                                          
        lists_of_halos_merging_at_current_z = np.where(z_merges==i)


        all_halos_merging_at_current_z=[]
        qvalsz = []
        m200s_list_main = []

        # collect halo objects of merging halos for the redhsift 'i'                                                                                   
        for list_of_halos in lists_of_halos_merging_at_current_z :
            halos_merging_at_current_z =np.array([])
            qvals = np.array([])
            m200s_list = np.array([])

            for merging_halo_object in list_of_halos:

                haloM =  h_merges[merging_halo_object][1:][0]

                halonums_over_life, ts_over_life = haloM.calculate_for_progenitors("halo_number()","t()")

                overlap = False


                #print( halonums_over_life, ts_over_life)                                                                                                    
                for h,t in zip(halonums_over_life,ts_over_life):

                    pair = [h,t]

                    if pair in list_of_selected_tuples:
                        print("overlap found")
                        overlap = True
                        

                    if (overlap == False):

                        list_of_selected_tuples.append(pair)

                if (overlap==True):
                    continue 

                print("FALSE OVERLAP")
                halos_merging_at_current_z = np.append(halos_merging_at_current_z, h_merges[merging_halo_object][1:])

                qvals = np.append(qvals,q_merges[merging_halo_object])

                if np.isin("M200c_DM",haloM.keys()):
                    #print(haloM["M200c"])                                                                                                                   
                    m200s_list = np.append(m200s_list,haloM["M200c_DM"])

                else:
                    m200s_list = np.append(m200s_list,0)

            all_halos_merging_at_current_z.append(halos_merging_at_current_z)

            qvalsz.append(qvals)

            m200s_list_main.append(m200s_list)


        merging_halos_grouped_by_z.append(all_halos_merging_at_current_z)
        qvalues_grouped_by_z.append(qvalsz)

        m200_groupd_by_z.append(m200s_list_main)

    return merging_halos_grouped_by_z, z_unique_values, qvalues_grouped_by_z, m200_groupd_by_z


def EuclideanDistance(xyz1,xyz2):

    x = xyz1[0] - xyz2[0]
    y = xyz1[1] - xyz2[1]
    z = xyz1[2] - xyz2[2]

    return np.sqrt( x**2 + y**2 + z**2 )


### Input Processing

haloname = str(sys.argv[1])

if len(str(haloname)) <= 8:
    DMOname = haloname+"_DMO"
    HYDROname = haloname+"_fiducial"

else: 
    HaloNameDecomp = haloname.split("_")
    DMOname = HaloNameDecomp[0]+"_DMO_"+HaloNameDecomp[2]
    HYDROname = HaloNameDecomp[0]+"_fiducial_"+HaloNameDecomp[2]
    haloname = HaloNameDecomp[0]


pynbody_path = '/scratch/dp101/shared/EDGE/'
# Finding best match in tangos db

# Start of crossreff 
tangos.core.init_db("/scratch/dp101/shared/EDGE/tangos/"+str(haloname)+".db")

## Get DMO data 
DMOsim = tangos.get_simulation(DMOname)
DMOMain = DMOsim.timesteps[-1].halos[0]
HaloNumsDMO = DMOMain.calculate_for_progenitors("halo_number()")[0][::-1]
RedsMainDMO = DMOMain.calculate_for_progenitors("z()")[0][::-1]

# create arrays that associate tangos timesteps with stored redshifts 
RedshiftsDMO= []
TimeStepIdxsDMO = []


for t in range(len(DMOsim.timesteps[:])):
    
    if len(DMOsim.timesteps[t].halos[:]) == 0:
        continue
    else: 
        RedshiftsDMO.append(DMOsim.timesteps[t].halos[0].calculate("z()"))
        TimeStepIdxsDMO.append(t)
        

RedshiftsDMO = np.asarray(RedshiftsDMO)
TimeStepIdxsDMO = np.asarray(TimeStepIdxsDMO)

# these two arrays should have the same length
print("DMO:",len(RedshiftsDMO),len(HaloNumsDMO),len(TimeStepIdxsDMO))

## Get HYDRO data 
HYDROsim = tangos.get_simulation(HYDROname)
HYDROMain = HYDROsim.timesteps[-1].halos[0]
HaloNumsHYDRO = HYDROMain.calculate_for_progenitors("halo_number()")[0][::-1]
RedshiftsTangosMainHYDRO = HYDROMain.calculate_for_progenitors("z()")[0][::-1]
RedshiftsHYDRO = []
TimeStepIdxsHYDRO = []
TimeStepsHYDRO = []
tstepidx = 0
for t in range(len(HYDROsim.timesteps[:])):
    
    if (len(HYDROsim.timesteps[t].halos[:]) == 0):
        
        continue
    else:
        RedshiftsHYDRO.append(HYDROsim.timesteps[t].halos[0].calculate("z()"))
        TimeStepIdxsHYDRO.append(t)
        TimeStepsHYDRO.append(str(HYDROsim.timesteps[t]))
    

TimeStepsHYDRO = np.asarray(TimeStepsHYDRO)
TimeStepIdxsHYDRO = np.asarray(TimeStepIdxsHYDRO)
RedshiftsHYDROAll = np.asarray(RedshiftsHYDRO)

# these should have the same lengths
print("HYDRO:",len(HaloNumsHYDRO),len(RedshiftsHYDRO))

#Processing Merger Tree 
MergerRedshiftsHYDRO, MergerRatiosHYDRO, MergerHaloObjectsHYDRO = get_mergers_of_major_progenitor(HYDROMain)
GroupedHalosHYDRO, GroupedRedshiftsHYDRO, GroupedMergerRatiosHYDRO,Groupedm200s = group_mergers(MergerRedshiftsHYDRO, MergerHaloObjectsHYDRO, MergerRatiosHYDRO)

idx_of_best_match_DMO = [np.argmin(RedshiftsDMO[np.where(RedshiftsDMO > zh)]) for zh in GroupedRedshiftsHYDRO]


print(len(idx_of_best_match_DMO) , len(GroupedRedshiftsHYDRO),":lengths arrays idx_of_best_match_DMO GroupedRedshiftsHYDRO")
#tstepidxsHYDRO = TimeStepIdxsHYDRO[np.asarray(idx_of_best_match_hydro)] 

# has length = len(GroupedRedshiftsHYDRO)
tstepidxsDMO = TimeStepIdxsDMO[np.asarray(idx_of_best_match_DMO)]


print(len(tstepidxsDMO) , len(GroupedRedshiftsHYDRO),":lengths arrays tstepidxsDMO GroupedRedshiftsHYDRO")

#print(idx_of_best_match_hydro)
hydrohalo_matched = []
dmohalo_matched = [] 
HydroHaloMstars = []

CentersMergingObjectsDMO = [] 
CentersMainHaloDMO = []
HnumsDMO = []
MassesDMO =[] 
VelDMO = []
TDMO = []
ZDMO = [] 
MassMainDMO =[]
R200DMO = [] 
EvolvedPositionsDMO = [] 
EvolvedTimesDMO = []
statusDMO = [] 


CentersMergingObjectsHYDRO =[]
CentersMainHaloHYDRO = []
HnumsHYDRO = []
MassHYDRO = []  
VelHYDRO = []
THYDRO = [] 
ZHYDRO = [] 
MassMainHydro = []
R200HYDRO = [] 
EvolvedPositionsHYDRO = [] 
EvolvedTimesHYDRO = []
statusHYDRO = [] 


print("starting loop")


pynbody.config["halo-class-priority"] = [pynbody.halo.hop.HOPCatalogue]

for z in range(len(GroupedRedshiftsHYDRO)):
    
    HYDROMergingHalosThisRedshift = GroupedHalosHYDRO[z][0]            
    
    print(z)

    if (len(HYDROMergingHalosThisRedshift) == 0): 
        print("no halos at this timestep")
        continue

    MergerTimestep = HYDROMergingHalosThisRedshift[0].timestep
    
    HYDROTimestepThisMerger = np.where(TimeStepsHYDRO == str(MergerTimestep))[0][0]
    
    HYDROhalonumidx = np.where(RedshiftsTangosMainHYDRO == RedshiftsHYDRO[HYDROTimestepThisMerger])[0][0]
    
    HYDROMainHaloThisRedshift = HYDROsim.timesteps[ TimeStepIdxsHYDRO[HYDROTimestepThisMerger] ].halos[ int(HaloNumsHYDRO[HYDROhalonumidx]) - 1 ]

    print(tstepidxsDMO[z])
    DMOhalonumidx = np.where(RedsMainDMO==RedshiftsDMO[idx_of_best_match_DMO[z]])[0][0]
    MainHaloDMOThisRedshift = DMOsim.timesteps[ tstepidxsDMO[z] ].halos[ int(HaloNumsDMO[DMOhalonumidx]) - 1 ]
    DMOHalosThisRedshift = list(DMOsim.timesteps[ tstepidxsDMO[z] ].halos[:])[:200]

    DMOHalosThisRedshift.remove(MainHaloDMOThisRedshift)
    dm_mass = []    

    output_num = str(DMOsim.timesteps[ tstepidxsDMO[z] ]).split("/")[-1][:12] 
    Prev_output_num = str(int(output_num) - 1).zfill(5)  
    Next_output_num = str(int(output_num) + 1).zfill(5)  

    simfnDMO = os.path.join(pynbody_path,DMOname,output_num)                                                         
    DMOParticles = pynbody.load(simfnDMO)                                                                            
    DMOParticles.physical_units()
    
    simfnDMOprev = os.path.join(pynbody_path,DMOname,Prev_output_num)                                                         
    DMOParticlesprev = pynbody.load(simfnDMOprev)                                                                            
    DMOParticlesprev.physical_units()
    
    simfnDMONext = os.path.join(pynbody_path,DMOname,Next_output_num)                                                         
    DMOParticlesNext = pynbody.load(simfnDMONext)                                                                            
    DMOParticlesNext.physical_units()
    #pynbody.analysis.halo.center(DMOParticles.halos()[int(findAHFhalonumINSITU(DMOname,output_num)) - 1])
    #MainParticles = DMOParticles.halos(halo_numbers='v1')[int(findAHFhalonumINSITU(DMOname,output_num)) ]
    MainParticles = DMOParticles.halos()[MainHaloDMOThisRedshift.calculate("halo_number()")]
    MainParticlesAtNextTime = DMOParticlesNext[np.isin(DMOParticlesNext['iord'],MainParticlesAtCurrentTime)]
    MainParticlesAtPrevTime = DMOParticlesprev.halos()[int(MainHaloDMOThisRedshift.calculate_for_progenitors("halo_number()")[0][1]) - 1]

    #print(MainParticles.properties)

    mergertimestep = [] 
    
    MainVelx = np.mean(MainParticles["vel"][:,0])
    MainVely = np.mean(MainParticles["vel"][:,1])
    MainVelz = np.mean(MainParticles["vel"][:,2])

    MainVelxNext = np.mean(MainParticles["vel"][:,0])
    MainVelyNext = np.mean(MainParticles["vel"][:,1])
    MainVelzNext = np.mean(MainParticles["vel"][:,2])

    MainVelxPrev = np.mean(MainParticles["vel"][:,0])
    MainVelyPrev = np.mean(MainParticles["vel"][:,1])
    MainVelzPrev = np.mean(MainParticles["vel"][:,2])

    
    for DMOhalo in DMOHalosThisRedshift:
            
        try:
            # Current snap 
            dm_mass.append(MainHaloDMOThisRedshift.calculate("M200c")/DMOhalo.calculate("M200c"))
                        
            vels = DMOParticles.halos()[int(DMOhalo.calculate("halo_number()")) - 1]["vel"]
            ParticlesAtCurrentTime = DMOParticles.halos()[int(DMOhalo.calculate("halo_number()")) - 1]['iord']
            MainParticlesAtCurrentTime = DMOParticles.halos()[int(MainHaloDMOThisRedshift.calculate("halo_number()")) - 1]['iord']
            
            MassesDMO.append(DMOhalo.calculate("M200c")) 
            HnumsDMO.append(DMOhalo.calculate("halo_number()"))
            MassMainDMO.append(MainHaloDMOThisRedshift.calculate("M200c"))
            R200DMO.append(MainHaloDMOThisRedshift.calculate("r200c"))
            
            CentersMergingObjectsDMO.append(DMOhalo["shrink_center"])
            CentersMainHaloDMO.append(MainHaloDMOThisRedshift["shrink_center"])
            
            TDMO.append(DMOhalo.calculate("t()"))
            ZDMO.append(DMOhalo.calculate("z()"))

            VelDMO.append([np.mean(vels[:,0]-MainVelx) , np.mean(vels[:,1]-MainVely), np.mean(vels[:,2] - MainVelz)])

            # Next Snap 
            
            
            ParticlesAtNextTime = DMOParticlesNext[np.isin(DMOParticlesNext['iord'],ParticlesAtCurrentTime)]

            vels = ParticlesAtNextTime["vel"]

            MassesDMO.append(DMOhalo.calculate("M200c")) 
            HnumsDMO.append(DMOhalo.calculate("halo_number()"))
            MassMainDMO.append(MainHaloDMOThisRedshift.calculate("M200c"))
            R200DMO.append(MainHaloDMOThisRedshift.calculate("r200c"))
            
            CentersMergingObjectsDMO.append(pynbody.analysis.halo.center(ParticlesAtNextTime,retcen=True))
            CentersMainHaloDMO.append(pynbody.analysis.halo.center(MainParticlesAtNextTime,retcen=True))
            
            TDMO.append(DMOhalo.calculate("t()"))
            ZDMO.append(DMOhalo.calculate("z()"))

            VelDMO.append([np.mean(vels[:,0]-MainVelxNext) , np.mean(vels[:,1]-MainVelyNext), np.mean(vels[:,2] - MainVelzNext)])

            
            # Previous Snapshot 
            
            
            ParticlesAtPrevTime = DMOParticlesprev.halos()[int(DMOhalo.calculate("halo_number()")[0][1]) - 1]

            vels = ParticlesAtPrevTime["vel"]
            
            MassesDMO.append(DMOhalo.calculate("M200c")) 
            HnumsDMO.append(DMOhalo.calculate("halo_number()"))
            MassMainDMO.append(MainHaloDMOThisRedshift.calculate("M200c"))
            R200DMO.append(MainHaloDMOThisRedshift.calculate("r200c"))
            
            CentersMergingObjectsDMO.append(pynbody.analysis.halo.center(ParticlesAtPrevTime,retcen=True))
            CentersMainHaloDMO.append(pynbody.analysis.halo.center(MainParticlesAtPrevTime,retcen=True))
            
            TDMO.append(DMOhalo.calculate("t()"))
            ZDMO.append(DMOhalo.calculate("z()"))
            VelDMO.append([np.mean(vels[:,0]-MainVelxPrev) , np.mean(vels[:,1]-MainVelyPrev), np.mean(vels[:,2] - MainVelzPrev)])



    
    
    print("loading in hydro data")
    # load in HYDRO data 
    outputHYDRO = str(HYDROsim.timesteps[ TimeStepIdxsHYDRO[HYDROTimestepThisMerger]  ]).split("/")[-1][:12]
    Prev_output_num = str(int(outputHYDRO) - 1).zfill(5)  
    Next_output_num = str(int(outputHYDRO) + 1).zfill(5)  
    
    simfnHYDRO = os.path.join(pynbody_path,HYDROname,outputHYDRO)
    HYDROParticles = pynbody.load(simfnHYDRO)
    HYDROMainHalo = HYDROParticles.halos()[int(HYDROMainHaloThisRedshift.calculate("halo_number()")) - 1 ]

    simfnDMOprev = os.path.join(pynbody_path,DMOname,Prev_output_num)                                                         
    DMOParticlesprev = pynbody.load(simfnDMOprev)                                                                            
    DMOParticlesprev.physical_units()
    
    simfnDMONext = os.path.join(pynbody_path,DMOname,Next_output_num)                                                         
    DMOParticlesNext = pynbody.load(simfnDMONext)                                                                            
    DMOParticlesNext.physical_units()

    
    HYDROMainParticlesAtNextTime =  HYDROParticlesNext[np.isin(HYDROParticlesNext['iord'],HYDROMainHalo)]
    HYDROMainParticlesAtPrevTime = HYDROParticlesprev.halos()[int(HYDROMainHaloThisRedshift.calculate_for_progenitors("halo_number()")[0][1]) - 1]

    HYDROParticles.physical_units()
    
    MainVelx = np.mean(HYDROMainHalo["vel"][:,0])
    MainVely = np.mean(HYDROMainHalo["vel"][:,1])
    MainVelz = np.mean(HYDROMainHalo["vel"][:,2])

    MainVelxNext = np.mean(HYDROMainParticlesAtNextTime["vel"][:,0])
    MainVelyNext = np.mean(HYDROMainParticlesAtNextTime["vel"][:,1])
    MainVelzNext = np.mean(HYDROMainParticlesAtNextTime["vel"][:,2])

    MainVelxPrev = np.mean(HYDROMainParticlesAtPrevTime["vel"][:,0])
    MainVelyPrev = np.mean(HYDROMainParticlesAtPrevTime["vel"][:,1])
    MainVelzPrev = np.mean(HYDROMainParticlesAtPrevTime["vel"][:,2])


    for MergingHYDROhalo in HYDROMergingHalosThisRedshift:
        print(MergingHYDROhalo)
        
        try:
            
            MassHYDRO.append(MergingHYDROhalo.calculate("M200c_DM"))
            CentersMergingObjectsHYDRO.append(MergingHYDROhalo["shrink_center"])
            CentersMainHaloHYDRO.append(HYDROMainHaloThisRedshift["shrink_center"])
            HnumsHYDRO.append(MergingHYDROhalo.calculate("halo_number()"))
            MassMainHydro.append(HYDROMainHaloThisRedshift["M200c_DM"])
            R200HYDRO.append(HYDROMainHaloThisRedshift.calculate("r200c"))
            MergingHYDROPynbody = HYDROParticles.halos()[int(MergingHYDROhalo.calculate("halo_number()")) - 1 ]

            THYDRO.append(MergingHYDROhalo.calculate("t()"))
            ZHYDRO.append(MergingHYDROhalo.calculate("z()"))

            VelHYDRO.append([np.mean(MergingHYDROPynbody["vel"][:,0] - MainVelx) , np.mean(MergingHYDROPynbody["vel"][:,1]-MainVely), np.mean(MergingHYDROPynbody["vel"][:,2]-MainVelz) ])
            '''
            ParticlesAtNextTime = DMOParticlesNext[np.isin(DMOParticlesNext['iord'],ParticlesAtCurrentTime)]

            vels = ParticlesAtNextTime["vel"]

            MassHYDRO.append(MergingHYDROhalo.calculate("M200c_DM"))
            HnumsHYDRO.append(MergingHYDROhalo.calculate("halo_number()"))
            MassMainHydro.append(HYDROMainHaloThisRedshift["M200c_DM"])
            R200HYDRO.append(HYDROMainHaloThisRedshift.calculate("r200c"))
            
            MergingHYDROPynbody = HYDROParticles.halos()[int(MergingHYDROhalo.calculate("halo_number()")) - 1 ]

            CentersMergingObjectsHYDRO.append(MergingHYDROhalo["shrink_center"])
            CentersMainHaloHYDRO.append(HYDROMainHaloThisRedshift["shrink_center"])

            
            CentersMergingObjectsDMO.append(pynbody.analysis.halo.center(ParticlesAtNextTime,retcen=True))
            CentersMainHaloDMO.append(pynbody.analysis.halo.center(MainParticlesAtNextTime,retcen=True))
            
            TDMO.append(DMOhalo.calculate("t()"))
            ZDMO.append(DMOhalo.calculate("z()"))

            VelDMO.append([np.mean(vels[:,0]-MainVelxNext) , np.mean(vels[:,1]-MainVelyNext), np.mean(vels[:,2] - MainVelzNext)])

            '''


        
        except Exception as e:
            print(e)
            continue 
            
            

print(len(MassHYDRO),len(HnumsHYDRO),len(CentersMergingObjectsHYDRO),len(CentersMainHaloHYDRO),len(VelHYDRO),len(THYDRO))

df = pd.DataFrame({"MergingCen":CentersMergingObjectsHYDRO,"MainCen":CentersMainHaloHYDRO,"Hnum":HnumsHYDRO,"M":MassHYDRO,"vel":VelHYDRO,"t":THYDRO,"z":ZHYDRO,"MassMain":MassMainHydro,"R200":R200HYDRO})
dfDMO = pd.DataFrame({"MergingCen":CentersMergingObjectsDMO,"MainCen":CentersMainHaloDMO,"Hnum":HnumsDMO,"M":MassesDMO,"vel":VelDMO,"t":TDMO,"z":ZDMO,"MassMain":MassMainDMO,"R200":R200DMO})


df.to_csv("dmo_hydro_crossreffs/HYDROcens_"+haloname+".csv")
dfDMO.to_csv("dmo_hydro_crossreffs/DMOcens_"+haloname+".csv")

