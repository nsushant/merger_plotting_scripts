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


'''
def calculate_Rizzo_energy_distance(X,Y):

    # Returns a measure of simmilarity between two distributions called the Energy Distance 
    # A lower energy distance = distributions are more simmilar (vice-versa)
    # citation: WIREs Comput Stat 2016, 8:27–38. doi: 10.1002/wics.1375 (see sec. TESTING FOR EQUAL DISTRIBUTIONS)
    
    XYMeanCdist = np.mean(cdist(X, Y))   # Spread between the two distributions of samples
    XXMeanCdist = np.mean(cdist(X, X))   # Spread within distribution X
    YYMeanCdist = np.mean(cdist(Y, Y))   # Spread within distribution Y
    
    return 2 *XYMeanCdist - XXMeanCdist - YYMeanCdist 
'''

    

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

                #halos_merging_at_current_z = np.append(halos_merging_at_current_z, h_merges[merging_halo_object][1:])                                       
                #print("len hmerge: ",len(h_merges[merging_halo_object][1:]))                                                                                
                #qvals = np.append(qvals,q_merges[merging_halo_object])                                                                                      
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

                #halonums_over_life, ts_over_life = haloM.calculate_for_progenitors("halo_number()","t()")                                                   

                #for h,t in zip(halonums_over_life[0],ts_over_life[0]):                                                                                      

                #if np.logical_not(np.isin(tuple(h,t),lists_of_halos_merging_at_current_z)):                                                                 

                #lists_of_halos_merging_at_current_z = np.append(lists_of_halos_merging_at_current_z,tuple(h,t))                                             


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
    simfnDMO = os.path.join(pynbody_path,DMOname,output_num)                                                         
    DMOParticles = pynbody.load(simfnDMO)                                                                            
    DMOParticles.physical_units()
    #pynbody.analysis.halo.center(DMOParticles.halos()[int(findAHFhalonumINSITU(DMOname,output_num)) - 1])
    #MainParticles = DMOParticles.halos(halo_numbers='v1')[int(findAHFhalonumINSITU(DMOname,output_num)) ]
    MainParticles = DMOParticles.halos()[MainHaloDMOThisRedshift.calculate("halo_number()")]
    #print(MainParticles.properties)

    MainVelx = np.mean(MainParticles["vel"][:,0])
    MainVely = np.mean(MainParticles["vel"][:,1])
    MainVelz = np.mean(MainParticles["vel"][:,2])
    for DMOhalo in DMOHalosThisRedshift:
            
        try:
            #AHFhalonumACC = findAHFhalonumACC(int(DMOhalo.calculate("halo_number()")),DMOname,output_num)
            #if len(AHFhalonumACC) == 0:
            #    continue 
            #MainVelx = MainParticles.properties["VXc"]
            #MainVely = MainParticles.properties["VYc"]
            #MainVelz = MainParticles.properties["VZc"]
            dm_mass.append(MainHaloDMOThisRedshift.calculate("M200c")/DMOhalo.calculate("M200c"))

            MassesDMO.append(DMOhalo.calculate("M200c")) 
            HnumsDMO.append(DMOhalo.calculate("halo_number()"))
            MassMainDMO.append(MainHaloDMOThisRedshift.calculate("M200c"))
            R200DMO.append(MainHaloDMOThisRedshift.calculate("r200c"))
            
            CentersMergingObjectsDMO.append(DMOhalo["shrink_center"])
            CentersMainHaloDMO.append(MainHaloDMOThisRedshift["shrink_center"])
            
            #AHFhalonumACC = findAHFhalonumACC(int(DMOhalo.calculate("halo_number()")),DMOname,output_num)
            
            vels = DMOParticles.halos()[int(DMOhalo.calculate("halo_number()")) - 1]["vel"]
            #merginghaloAHF = DMOParticles.halos(halo_numbers='v1')[AHFhalonumACC] 
            
            TDMO.append(DMOhalo.calculate("t()"))
            ZDMO.append(DMOhalo.calculate("z()"))
            #VelDMO.append([np.mean(merginghaloAHF["vel"][:,0])-MainVelx , np.mean(merginghaloAHF["vel"][:,1])-MainVely, np.mean(merginghaloAHF["vel"][:,0]) - MainVelz])
            VelDMO.append([np.mean(vels[:,0]-MainVelx) , np.mean(vels[:,1]-MainVely), np.mean(vels[:,2] - MainVelz)])
            
        except:
            dm_mass.append(0)
            continue


    
    print("loading in hydro data")
    # load in HYDRO data 
    outputHYDRO = str(HYDROsim.timesteps[ TimeStepIdxsHYDRO[HYDROTimestepThisMerger]  ]).split("/")[-1][:12]
    simfnHYDRO = os.path.join(pynbody_path,HYDROname,outputHYDRO)
    HYDROParticles = pynbody.load(simfnHYDRO)
    #pynbody.analysis.halo.center(HYDROParticles.halos()[int(HaloNumsHYDRO[HYDROhalonumidx]) - 1])
    #HYDROMainHalo = HYDROParticles.halos()[int(HaloNumsHYDRO[HYDROhalonumidx]) - 1]
    HYDROMainHalo = HYDROParticles.halos(halo_numbers='v1')[int(findAHFhalonumINSITU(HYDROname,outputHYDRO))]
    HYDROParticles.physical_units()
    
    #ParticlesLoadedIn = False
    
    #MainParticles = DMOParticles.halos()[int(findAHFhalonumINSITU(DMOname,output_num)) - 1]

    #MainVelx = HYDROMainHalo.properties["VXc"]
    #MainVely = HYDROMainHalo.properties["VYc"]
    #MainVelz = HYDROMainHalo.properties["VZc"]
    MainVelx = np.mean(HYDROMainHalo["vel"][:,0])
    MainVely = np.mean(HYDROMainHalo["vel"][:,1])
    MainVelz = np.mean(HYDROMainHalo["vel"][:,2])

    for MergingHYDROhalo in HYDROMergingHalosThisRedshift:
        print(MergingHYDROhalo)
        
        try:
            
            #THYDRO.append(MergingHYDROhalo.calculate("t()"))
            #ZHYDRO.append(MergingHYDROhalo.calculate("z()"))
            MassHYDRO.append(MergingHYDROhalo.calculate("M200c_DM"))
            CentersMergingObjectsHYDRO.append(MergingHYDROhalo["shrink_center"])
            CentersMainHaloHYDRO.append(HYDROMainHaloThisRedshift["shrink_center"])
            HnumsHYDRO.append(MergingHYDROhalo.calculate("halo_number()"))
            MassMainHydro.append(HYDROMainHaloThisRedshift["M200c_DM"])
            #AHFhalonumACC = findAHFhalonumACC(int(MergingHYDROhalo.calculate("halo_number()")),HYDROname,outputHYDRO)
            R200HYDRO.append(HYDROMainHaloThisRedshift.calculate("r200c"))
            # Use AHF Halo prop VelCX.
            #print("halonum:",AHFhalonumACC)
            MergingHYDROPynbody = HYDROParticles.halos()[int(MergingHYDROhalo.calculate("halo_number()")) - 1 ]

            THYDRO.append(MergingHYDROhalo.calculate("t()"))
            ZHYDRO.append(MergingHYDROhalo.calculate("z()"))
            #HYDROMainHalo.properties["VelCX"]
            VelHYDRO.append([np.mean(MergingHYDROPynbody["vel"][:,0] - MainVelx) , np.mean(MergingHYDROPynbody["vel"][:,1]-MainVely), np.mean(MergingHYDROPynbody["vel"][:,2]-MainVelz) ])
            
            #THYDRO.append(MergingHYDROhalo.calculate("t()"))
            #ZHYDRO.append(MergingHYDROhalo.calculate("z()"))
        except Exception as e:
            print(e)
            continue 
            
            
        '''

        try: 
            MergingHYDROhalo["M200c_stars"]
            
            if MergingHYDROhalo["M200c_stars"] == 0: 
                print("No Mstar")
                continue 
            
        except Exception as e: 
            print(e)
            continue

        
        try:
            m200MergingHYDROhalo = HYDROMainHaloThisRedshift["M200c"]/MergingHYDROhalo["M200c"]
        
            print("added")

        except Exception as er:
            print(er)
            continue
            
            
        # sorts mass difference in M200 in ascending order    
        closest_mass_match = np.argsort(np.abs(np.asarray(dm_mass) - m200MergingHYDROhalo))[:5]                                  
        print("tangos ended")
        ### Tangos part ends and energy distance calculation starts 
        #HYDRO halo 6D array
        MergingHYDROHaloNumber = MergingHYDROhalo.calculate("halo_number()")
        MergingHYDROHaloParticles = HYDROParticles.halos()[int(MergingHYDROHaloNumber)-1]
        MergingHYDROHaloParticles = MergingHYDROHaloParticles[MergingHYDROHaloParticles["r"] < 2]

        px = MergingHYDROHaloParticles.dm["vel"][:,0]*MergingHYDROHaloParticles.dm["mass"]
        py = MergingHYDROHaloParticles.dm["vel"][:,1]*MergingHYDROHaloParticles.dm["mass"]
        pz = MergingHYDROHaloParticles.dm["vel"][:,2]*MergingHYDROHaloParticles.dm["mass"]
        
        PhaseArrayHydro = np.stack((MergingHYDROHaloParticles.d['x'], MergingHYDROHaloParticles.d['y'], MergingHYDROHaloParticles.d['z'], px, py, pz), axis=1)

        # Load in DMO data
        if ParticlesLoadedIn == False:          
            # load in DMO pynbody data
            output_num = str(DMOsim.timesteps[ tstepidxsDMO[z] ]).split("/")[-1][:12]
            simfnDMO = os.path.join(pynbody_path,DMOname,output_num)
            DMOParticles = pynbody.load(simfnDMO)
            DMOParticles.physical_units()
            ParticlesLoadedIn= True

            print("loaded in DMO")
        

        PhaseArraysDMO = []
        for MassMatch in closest_mass_match:
            MassMatchedDMOHalo = DMOHalosThisRedshift[MassMatch]
            HaloNumMassMatchedDMOHalo = MassMatchedDMOHalo.calculate("halo_number()")
            MassMatchedDMOHaloParticles = DMOParticles.halos()[int(HaloNumMassMatchedDMOHalo) - 1]
            MassMatchedDMOHaloParticles = MassMatchedDMOHaloParticles[MassMatchedDMOHaloParticles['r'] < 2]
            pxd = MassMatchedDMOHaloParticles["vel"][:,0]*MassMatchedDMOHaloParticles["mass"]
            pyd = MassMatchedDMOHaloParticles["vel"][:,1]*MassMatchedDMOHaloParticles["mass"]
            pzd = MassMatchedDMOHaloParticles["vel"][:,2]*MassMatchedDMOHaloParticles["mass"]
            PhaseArraysDMO.append(np.stack((MassMatchedDMOHaloParticles["x"],MassMatchedDMOHaloParticles["y"],MassMatchedDMOHaloParticles["z"],pxd,pyd,pzd),axis=1))
            

        EnergyDistances = np.array([])

        for PhaseArrayDMO in PhaseArraysDMO: 

            EnergyDistanceHalo = calculate_Rizzo_energy_distance(PhaseArrayDMO,PhaseArrayHydro)
            EnergyDistances = np.append(EnergyDistances,EnergyDistanceHalo)

        
        best_match_2_fold =  closest_mass_match[np.argmin(EnergyDistances)]
        

        hydrohalo_matched.append(MergingHYDROhalo)
        HydroHaloMstars.append(MergingHYDROhalo["M200c_stars"])
        dmohalo_matched.append(DMOHalosThisRedshift[best_match_2_fold])
        #dmohalo_matched.append(d.timesteps[idx_of_best_match[z]].halos[closest_mass_match[0]])
        '''
#print(hydrohalo_matched)
#print(dmohalo_matched)

#df = pd.DataFrame({"halo":dmohalo_matched,"mstar":HydroHaloMstars,"hydrohalo":hydrohalo_matched})                                                                  

#print(df)
#df.to_csv("dmo_hydro_crossreffs/TwoFoldCrossreff_WithPynbody_"+DMOname+".csv")                    


print(len(MassHYDRO),len(HnumsHYDRO),len(CentersMergingObjectsHYDRO),len(CentersMainHaloHYDRO),len(VelHYDRO),len(THYDRO))

df = pd.DataFrame({"MergingCen":CentersMergingObjectsHYDRO,"MainCen":CentersMainHaloHYDRO,"Hnum":HnumsHYDRO,"M":MassHYDRO,"vel":VelHYDRO,"t":THYDRO,"z":ZHYDRO,"MassMain":MassMainHydro,"R200":R200HYDRO})
dfDMO = pd.DataFrame({"MergingCen":CentersMergingObjectsDMO,"MainCen":CentersMainHaloDMO,"Hnum":HnumsDMO,"M":MassesDMO,"vel":VelDMO,"t":TDMO,"z":ZDMO,"MassMain":MassMainDMO,"R200":R200DMO})


df.to_csv("dmo_hydro_crossreffs/HYDROcens_"+haloname+".csv")
dfDMO.to_csv("dmo_hydro_crossreffs/DMOcens_"+haloname+".csv")

