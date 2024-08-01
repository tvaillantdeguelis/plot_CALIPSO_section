#!/bin/bash

# This program launches a list of case studies to plot.

# Run plot_calipso_section.py
plot_calipso_section () {
       jobname="plot_"$1"_"$3"_$4"
       echo -e "jobname=$jobname"
       sbatch --job-name=$jobname \
              --error=./sbatch_out/${jobname}.e \
              --output=./sbatch_out/${jobname}.o \
              --export=GRANULE_DATE="$1",SLICE_START_END_TYPE="$2",SLICE_START="$3",SLICE_END="$4",CASE_STUDY_NAME="$5" plot_calipso_section.sbatch
}

case_study_name="Polar stratospheric clouds"
granule_date="2008-07-17T19-15-43ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=83.89 # profindex or longitude
slice_end=7.65 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Artic snow/ice surfaces"
granule_date="2017-02-10T23-46-03ZD"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=-167.35 # profindex or longitude
slice_end=146.90 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Strongly depolarizing cirrus"
granule_date="2017-02-10T00-41-35ZD"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=-154.50 # profindex or longitude
slice_end=-169.64 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Strongly depolarizing low cloud"
granule_date="2017-02-10T18-02-57ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=120.78 # profindex or longitude
slice_end=108.36 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="PyroCb smokes vs non-depolarizing smoke"
granule_date="2017-08-20T00-58-21ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=30.45 # profindex or longitude
slice_end=10.81 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Dust vs Smoke (Night)"
granule_date="2019-07-15T00-58-53ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=16.00 # profindex or longitude
slice_end=5.00 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Dust vs Smoke (Day)"
granule_date="2019-07-15T13-15-00ZD"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=8.30 # profindex or longitude
slice_end=-2.62 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Smoke from Australian bushfires"
granule_date="2009-02-10T12-33-03ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=-168.15 # profindex or longitude
slice_end=178.68 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Smoke over cirrus"
granule_date="2012-05-12T11-31-08ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=-132.29 # profindex or longitude
slice_end=-148.26 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Cloud chaos: HOI ROI H2O"
granule_date="2007-04-10T04-21-17ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=-15.34 # profindex or longitude
slice_end=-39.11 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Liquid and ice clouds"
granule_date="2016-06-15T09-59-23ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=-111.76 # profindex or longitude
slice_end=-126.11 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Complex clouds"
granule_date="2008-08-22T12-13-19ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=-137.23 # profindex or longitude
slice_end=-157.47 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

# case_study_name="Yorks et al. (2011) opaque cirrus"
# granule_date="2006-08-11T07-52-34ZN"
# slice_start_end_type='longitude' # 'profindex' or 'longitude'
# slice_start=-87 # profindex or longitude
# slice_end=-88.6 # profindex or longitude
# plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Dust and clouds"
granule_date="2016-07-04T12-55-03ZD"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=11.70 # profindex or longitude
slice_end=0.77 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Saharan dust storm"
granule_date="2018-06-18T02-53-21ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=-5.13 # profindex or longitude
slice_end=-19.47 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"

case_study_name="Antarctica surface"
granule_date="2010-06-14T14-13-41ZN"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=158.87 # profindex or longitude
slice_end=64.36 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"


case_study_name="Dust plume"
granule_date="2014-06-23T15-11-54ZD"
slice_start_end_type='longitude' # 'profindex' or 'longitude'
slice_start=-22.53 # profindex or longitude
slice_end=-33.48 # profindex or longitude
plot_calipso_section $granule_date $slice_start_end_type $slice_start $slice_end "$case_study_name"
