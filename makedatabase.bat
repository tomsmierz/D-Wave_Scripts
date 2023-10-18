@echo off
setlocal enabledelayedexpansion

set Topologies=Pegasus Zephyr
set PegasusSizes=4 8 12 16
set ZephyrSizes=2 3 4
set PegasusTypes=RAU RCO AC3 CBFM-P
set ZephyrTypes=RAU RCO AC3
set desiredLength=3

for %%T in (%Topologies%) do (
	if %%T==Pegasus (
		for %%P in (%PegasusSizes%) do (
			for %%C in (%PegasusTypes%) do (
				set "source=C:\Users\tsmierzchalski\PycharmProjects\D-Wave_Scripts\instances\pegasus_random\P%%P\%%C"
				set "agg_source=C:\Users\tsmierzchalski\PycharmProjects\D-Wave_Scripts\energies\pegasus_random_aggregated\P%%P\%%C"
				for /L %%N in (1,1,100) do (			
					set "targetDirectory=C:\Users\tsmierzchalski\%%T\P%%P\%%C\%%N"
					if %%N lss 10 (
							set name=00%%N
						) else if %%N lss 100 (
							set name=0%%N
						) else (
							set name=%%N
						)
					if not exist "!targetDirectory!" (
						mkdir "!targetDirectory!"
					)
					if not exist !targetDirectory!\P%%P_%%C_SG_%%N.txt (
						copy !source!\!name!_sg.txt !targetDirectory!
						ren !targetDirectory!\!name!_sg.txt P%%P_%%C_SG_%%N.txt
					) else (
						echo file P%%P_%%C_SG_%%N.txt exist
					)
					if not exist !targetDirectory!\P%%P_%%C_DV_%%N.pkl (
						
						copy !source!\!name!_dv.pkl !targetDirectory!
						ren !targetDirectory!\!name!_dv.pkl P%%P_%%C_DV_%%N.pkl
					) else (
						echo file P%%P_%%C_DV_%%N.pkl exist
					)
					if %%P==4 (
						if not exist !targetDirectory!\solutions (
							mkdir !targetDirectory!\solutions
						)
						if not exist !targetDirectory!\solutions\DWAVE_full.csv (
							copy !agg_source!\!name!.csv !targetDirectory!\solutions
							ren !targetDirectory!\solutions\!name!.csv DWAVE_full.csv
						)
					)
					if %%P==8 (
						if not exist !targetDirectory!\solutions (
							mkdir !targetDirectory!\solutions
						)
						if not exist !targetDirectory!\solutions\DWAVE_full.csv (
							copy !agg_source!\!name!.csv !targetDirectory!\solutions
							ren !targetDirectory!\solutions\!name!.csv DWAVE_full.csv
						)
					)
					
				)
			)
		)
	) else (
		for %%Z in (%ZephyrSizes%) do (
			for %%C in (%ZephyrTypes%) do (
				set "source=C:\Users\tsmierzchalski\PycharmProjects\D-Wave_Scripts\instances\zephyr_random\Z%%Z\%%C"
				for /L %%N in (1,1,100) do (			
					set "targetDirectory=C:\Users\tsmierzchalski\%%T\Z%%Z\%%C\%%N"
					if not exist "!targetDirectory!" (
						mkdir "!targetDirectory!"
					)
					if not exist !targetDirectory!\Z%%Z_%%C_SG_%%N.txt (
						if %%N lss 10 (
							set name=00%%N
						) else if %%N lss 100 (
							set name=0%%N
						) else (
							set name=%%N
						)
						copy !source!\!name!_sg.txt !targetDirectory!
						ren !targetDirectory!\!name!_sg.txt Z%%Z_%%C_SG_%%N.txt
					) else (
						echo file Z%%Z_%%C_SG_%%N.txt exist
					)		
					if not exist !targetDirectory!\Z%%Z_%%C_DV_%%N.pkl (
						if %%N lss 10 (
							set name=00%%N
						) else if %%N lss 100 (
							set name=0%%N
						) else (
							set name=%%N
						)
						copy !source!\!name!_dv.pkl !targetDirectory!
						ren !targetDirectory!\!name!_dv.pkl Z%%Z_%%C_DV_%%N.pkl
					) else (
						echo file Z%%Z_%%C_DV_%%N.pkl exist
					)
				)
			)
		)
	)
)
