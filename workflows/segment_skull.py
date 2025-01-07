#!/usr/bin/env python3
"""
    Non humain primates anatomical segmentation pipeline based ANTS

    Adapted in Nipype from an original pipelin of Kepkee Loh wrapped.

    Description
    --------------
    TODO :/

    Arguments
    -----------
    -data:
        Path to the BIDS directory that contain subjects' MRI data.

    -out:
        Nipype's processing directory.
        It's where all the outputs will be saved.

    -subjects:
        IDs list of subjects to process.

    -ses
        session (leave blank if None)

    -params
        json parameter file; leave blank if None

    Example
    ---------
    python segment_pnh.py -data [PATH_TO_BIDS] -out ../local_tests/ -subjects Elouk

    Requirements
    --------------
    This workflow use:
        - ANTS
        - AFNI
        - FSL
"""

# Authors : David Meunier (david.meunier@univ-amu.fr)
#           Bastien Cagna (bastien.cagna@univ-amu.fr)
#           Kepkee Loh (kepkee.loh@univ-amu.fr)
#           Julien Sein (julien.sein@univ-amu.fr)
#           Adam Sghari (adam.sghari@etu.univ-amu.fr)

import os
import os.path as op

import argparse
import json
import pprint

import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

import nipype.interfaces.fsl as fsl

from nipype.interfaces.niftyreg.regutils import RegResample

from macapype.pipelines.full_pipelines import (
    create_full_spm_subpipes,
    create_full_ants_subpipes,
    create_full_T1_ants_subpipes,)

from macapype.utils.utils_bids import (
    create_datasource,
    create_datasource_indiv_params,
    create_datasink)

from macapype.utils.utils_tests import load_test_data, format_template

from macapype.utils.utils_params import update_params

from macapype.utils.misc import show_files, get_first_elem, parse_key

from macapype.pipelines.rename import rename_all_brain_derivatives


from skullTo3d.pipelines.angio_pipe import (
    create_angio_pipe, create_quick_angio_pipe,
    create_autonomous_quick_angio_pipe)

from skullTo3d.pipelines.skull_pipe import (
    create_skull_petra_pipe,
    create_autonomous_skull_petra_pipe,
    create_skull_ct_pipe,
    create_skull_t1_pipe)

from skullTo3d.pipelines.rename import (
    rename_all_skull_petra_derivatives,
    rename_all_skull_t1_derivatives,
    rename_all_skull_ct_derivatives, rename_all_angio_derivatives)


fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
##########################################################################

def create_main_workflow(data_dir, process_dir, soft, species, subjects,
                         sessions, brain_dt, skull_dt, acquisitions,
                         reconstructions, params_file, indiv_params_file,
                         mask_file, template_path, template_files, nprocs,
                         reorient, deriv, pad,
                         wf_name="macapype"):

    # macapype_pipeline
    """ Set up the segmentatiopn pipeline based on ANTS

    Arguments
    ---------
    data_path: pathlike str
        Path to the BIDS directory that contains anatomical images

    out_path: pathlike str
        Path to the ouput directory (will be created if not alredy existing).
        Previous outputs maybe overwritten.

    soft: str
        Indicate which analysis should be launched; so for, only spm and ants
        are accepted; can be extended

    subjects: list of str (optional)
        Subject's IDs to match to BIDS specification (sub-[SUB1], sub-[SUB2]...)

    sessions: list of str (optional)
        Session's IDs to match to BIDS specification (ses-[SES1], ses-[SES2]...)

    acquisitions: list of str (optional)
        Acquisition name to match to BIDS specification (acq-[ACQ1]...)

    reconstructions: list of str (optional)
        Reconstructions name to match to BIDS specification (rec-[ACQ1]...)

    indiv_params_file: path to a JSON file
        JSON file that specify some parameters of the pipeline,
        unique for the subjects/sessions.

    params_file: path to a JSON file
        JSON file that specify some parameters of the pipeline.

    nprocs: integer
        number of processes that will be launched by MultiProc


    Returns
    -------
    workflow: nipype.pipeline.engine.Workflow


    """

    if brain_dt is not None:
        brain_dt = [dt.lower() for dt in brain_dt]
    else:
        brain_dt = []

    skull_dt = [dt.lower() for dt in skull_dt]

    print("brain_dt: ", brain_dt)
    print("skull_dt: ", skull_dt)

    soft = soft.lower()

    ssoft = soft.split("_")

    new_ssoft = ssoft.copy()

    if 'test' in ssoft:
        new_ssoft.remove('test')

    if 'prep' in ssoft:
        new_ssoft.remove('prep')

    if 'noseg' in ssoft:
        new_ssoft.remove('noseg')

    if 'native' in ssoft:
        new_ssoft.remove('native')

    if 'template' in ssoft:
        new_ssoft.remove('template')

    if 'robustreg' in ssoft:
        new_ssoft.remove('robustreg')

    soft = "_".join(new_ssoft)

    print("soft: ", soft)
    print("ssoft: ", ssoft)

    # formating args
    data_dir = op.abspath(data_dir)

    process_dir = op.abspath(process_dir)

    try:
        os.makedirs(process_dir)

    except OSError:
        print("process_dir {} already exists".format(process_dir))

    # params
    if params_file is None:

        # species
        if species is not None:

            species = species.lower()

            rep_species = {"marmoset": "marmo",
                           "marmouset": "marmo",
                           "chimpanzee": "chimp"}

            if species in list(rep_species.keys()):
                species = rep_species[species]

            list_species = ["macaque", "marmo", "baboon", "chimp"]

            assert species in list_species, \
                "Error, species {} should in the following list {}".format(
                    species, list_species)

            package_directory = op.dirname(op.abspath(__file__))

            params_file = "{}/params_segment_{}_{}.json".format(
                package_directory, species, soft)

        else:
            print("Error, no -params or no -species was found (one or the \
                other is mandatory)")
            exit(-1)

        print("Using default params file:", params_file)

        params, indiv_params, extra_wf_name = update_params(
            ssoft=ssoft, subjects=subjects, sessions=sessions,
            params_file=params_file, indiv_params_file=indiv_params_file)

    else:

        # format for relative path
        params_file = op.abspath(params_file)

        # params
        assert op.exists(params_file), "Error with file {}".format(
            params_file)

        print("Using orig params file:", params_file)

        params = json.load(open(params_file))

        extra_wf_name = "_orig"
        indiv_params = {}

        # indiv_params
        if indiv_params_file is not None:

            # format for relative path
            indiv_params_file = op.abspath(indiv_params_file)

            assert op.exists(indiv_params_file), "Error with file {}".format(
                indiv_params_file)
            indiv_params = json.load(open(indiv_params_file))

        print("Using indiv_params:", indiv_params)

    # modifying if reorient
    if reorient is not None:
        print("reorient: ", reorient)

        if "skull_petra_pipe" in params.keys():
            params["skull_petra_pipe"]["avg_reorient_pipe"] = {
                "reorient":
                    {"origin": reorient, "deoblique": True}}

        if "short_preparation_pipe" in params.keys():
            params["short_preparation_pipe"]["avg_reorient_pipe"] = {
                "reorient":
                    {"origin": reorient, "deoblique": True}}

    pprint.pprint(params)

    # Workflow
    wf_name += extra_wf_name

    # soft
    wf_name += "_{}".format(soft)

    if 't1' in skull_dt:
        wf_name += "_t1"

    if 'petra' in skull_dt:
        wf_name += "_petra"

    if 'angio' in skull_dt:
        wf_name += "_angio"

    if 'ct' in skull_dt:
        wf_name += "_CT"

    if len(brain_dt) != 0:
        wf_name += "_segbrain"

    if 't1' in brain_dt:
        wf_name += "_t1"

    if 't2' in brain_dt:
        wf_name += "_t2"

    if mask_file is not None:
        wf_name += "_mask"

    if "spm" in ssoft or "spm12" in ssoft or "ants" in ssoft:
        print("Segmenting brain, default is t1 based")

        # adding forced space
        if "spm" in ssoft or "spm12" in ssoft:
            if 'native' in ssoft:
                wf_name += "_native"

        elif "ants" in ssoft:
            if "template" in ssoft:
                wf_name += "_template"

        # params_template
        if template_path is not None:

            # format for relative path
            template_path = op.abspath(template_path)

            assert os.path.exists(template_path), "Error, template_path {} do not exists".format(template_path)

            print(template_files)

            params_template = {}

            assert len(template_files) > 1, "Error, template_files unspecified {}".format(template_files)

            template_head = os.path.join(template_path,template_files[0])
            assert os.path.exists(template_head), "Could not find template_head {}".format(template_head)
            params_template["template_head"] = template_head

            template_brain = os.path.join(template_path,template_files[1])
            assert os.path.exists(template_brain), "Could not find template_brain {}".format(template_brain)
            params_template["template_brain"] = template_brain

            if len(template_files) == 3:

                template_seg = os.path.join(template_path,template_files[2])
                assert os.path.exists(template_seg), "Could not find template_seg {}".format(template_seg)
                params_template["template_seg"] = template_seg

            elif len(template_files) == 5:

                template_gm = os.path.join(template_path,template_files[2])
                assert os.path.exists(template_gm), "Could not find template_gm {}".format(template_gm)
                params_template["template_gm"] = template_gm

                template_wm = os.path.join(template_path,template_files[3])
                assert os.path.exists(template_wm), "Could not find template_wm {}".format(template_wm)
                params_template["template_wm"] = template_wm

                template_csf = os.path.join(template_path,template_files[4])
                assert os.path.exists(template_csf), "Could not find template_csf {}".format(template_csf)
                params_template["template_csf"] = template_csf

            else:
                print("Unknown template_files format, should be 3 or 5 files")
                exit(-1)

            params_template_stereo = params_template

        else:
            ### use template from params
            assert ("general" in params.keys() and \
                "template_name" in params["general"].keys()), \
                    "Error, the params.json should contains a general/template_name"

            template_name = params["general"]["template_name"]

            if "general" in params.keys() and "my_path" in params["general"].keys():
                my_path = params["general"]["my_path"]
            else:
                my_path = ""

            template_dir = load_test_data(template_name, path_to = my_path)
            params_template = format_template(template_dir, template_name)

            if "template_stereo_name" in params["general"].keys():

                template_stereo_name = params["general"]["template_stereo_name"]
                template_stereo_dir = load_test_data(template_stereo_name, path_to = my_path)
                params_template_stereo = format_template(template_stereo_dir, template_stereo_name)

            else:
                params_template_stereo = params_template

        print (params_template)

    else:
        print(f"error with {ssoft}, should be among [spm12, spm, ants])")

    # main_workflow
    main_workflow = pe.Workflow(name= wf_name)

    main_workflow.base_dir = process_dir

    if "template" in ssoft:
        space = "template"

    else:
        space = "native"


    # which soft is used
    if "spm" in ssoft or "spm12" in ssoft:
        segment_brain_pipe = create_full_spm_subpipes(
            params_template=params_template,
            params_template_stereo=params_template_stereo,
            params=params, pad=pad, space=space)

    elif "ants" in ssoft:
        if "t1" in brain_dt and 't2' in brain_dt:
            segment_brain_pipe = create_full_ants_subpipes(
                params_template=params_template,
                params_template_stereo=params_template_stereo,
                params=params, mask_file=mask_file, space=space, pad=pad)

        elif "t1" in brain_dt:
            segment_brain_pipe = create_full_T1_ants_subpipes(
                params_template=params_template,
                params_template_stereo=params_template_stereo,
                params=params, space=space, pad=pad)

    # list of all required outputs
    output_query = {}

    # T1 (mandatory, always added)
    # T2 is optional, if "_T1" is added in the -soft arg
    if 't1' in brain_dt or 't1' in skull_dt :
        output_query['T1'] = {
            "datatype": "anat", "suffix": "T1w",
            "extension": ["nii", ".nii.gz"]}

    if 't2' in brain_dt:
        output_query['T2'] = {
            "datatype": "anat", "suffix": "T2w",
            "extension": ["nii", ".nii.gz"]}

    if 'petra' in skull_dt:
        output_query['PETRA'] = {
            "datatype": "anat", "suffix": ["PDw", "petra"],
            "extension": ["nii", ".nii.gz"]}

    if 'ct' in skull_dt:
        output_query['CT'] = {
            "datatype": "anat", "suffix": "T2star",
            "acquisition": "CT",
            "extension": ["nii", ".nii.gz"]}

    if 'angio' in skull_dt:
        output_query['ANGIO'] = {
            "datatype": "anat", "suffix": "angio",
            "extension": ["nii", ".nii.gz"]}

    # indiv_params
    if indiv_params:
        print("Using indiv params")
        datasource = create_datasource_indiv_params(
            output_query, data_dir, indiv_params, subjects, sessions,
            acquisitions, reconstructions)

        if "spm" in ssoft or "spm12" in ssoft or "ants" in ssoft:
            main_workflow.connect(datasource, "indiv_params",
                                  segment_brain_pipe, 'inputnode.indiv_params')
    else:
        datasource = create_datasource(
            output_query, data_dir, subjects,  sessions, acquisitions,
            reconstructions)

    if "spm" in ssoft or "spm12" in ssoft or "ants" in ssoft:
        if "t1" in brain_dt:
            main_workflow.connect(datasource, 'T1',
                                segment_brain_pipe, 'inputnode.list_T1')

        if "t2" in brain_dt:
            main_workflow.connect(datasource, 'T2',
                                segment_brain_pipe, 'inputnode.list_T2')

        elif "t1" in brain_dt and "spm" in ssoft:
            # cheating using T2 as T1
            main_workflow.connect(datasource, 'T1',
                                segment_brain_pipe, 'inputnode.list_T2')

    if "petra" in skull_dt and "skull_petra_pipe" in params.keys():
        print("Found skull_petra_pipe")

        if len(brain_dt):

            skull_petra_pipe = create_skull_petra_pipe(
                params=parse_key(params, "skull_petra_pipe"))

            if "t1" in brain_dt and "t2" in brain_dt:
                # optimal pipeline, use T2
                main_workflow.connect(segment_brain_pipe,
                                      "outputnode.native_T2",
                                      skull_petra_pipe, 'inputnode.native_img')

            elif "t1" in brain_dt:
                main_workflow.connect(segment_brain_pipe,
                                      "outputnode.native_T1",
                                      skull_petra_pipe, 'inputnode.native_img')

            main_workflow.connect(segment_brain_pipe,
                                  "outputnode.stereo_padded_T1",
                                  skull_petra_pipe, 'inputnode.stereo_T1')

            main_workflow.connect(segment_brain_pipe,
                                  "outputnode.native_to_stereo_trans",
                                  skull_petra_pipe,
                                  'inputnode.native_to_stereo_trans')
        else:
            print("No brain segmentation")
            skull_petra_pipe = create_autonomous_skull_petra_pipe(
                params=parse_key(params, "skull_petra_pipe"))

        # all remaining connection
        main_workflow.connect(datasource, ('PETRA', show_files),
                              skull_petra_pipe, 'inputnode.petra')

        if indiv_params:
            main_workflow.connect(datasource, "indiv_params",
                                  skull_petra_pipe, 'inputnode.indiv_params')

        if pad and space == "native":

            # output node
            outputnode = pe.Node(
                niu.IdentityInterface(
                    fields=["native_petra_skull_mask",
                            "native_robustpetra_skull_mask",
                            "native_petra_head_mask"]),
                name='outputnode')

            if "short_preparation_pipe" in params.keys():
                if "crop_T1" in params["short_preparation_pipe"].keys():
                    pass
                else:
                    print("Using reg_aladin transfo to pad skull_mask back")

                    pad_petra_skull_mask = pe.Node(RegResample(inter_val="NN"),
                                                   name="pad_petra_skull_mask")

                    main_workflow.connect(
                        skull_petra_pipe, "outputnode.petra_skull_mask",
                        pad_petra_skull_mask, "flo_file")

                    main_workflow.connect(
                        segment_brain_pipe, "outputnode.native_T1",
                        pad_petra_skull_mask, "ref_file")

                    main_workflow.connect(
                        segment_brain_pipe,
                        "outputnode.stereo_to_native_trans",
                        pad_petra_skull_mask, "trans_file")

                    print("Using reg_aladin transfo to pad head_mask back")

                    pad_petra_head_mask = pe.Node(RegResample(inter_val="NN"),
                                                  name="pad_petra_head_mask")

                    main_workflow.connect(skull_petra_pipe,
                                          "outputnode.petra_head_mask",
                                          pad_petra_head_mask, "flo_file")

                    main_workflow.connect(segment_brain_pipe,
                                          "outputnode.native_T1",
                                          pad_petra_head_mask, "ref_file")

                    main_workflow.connect(segment_brain_pipe,
                                          "outputnode.stereo_to_native_trans",
                                          pad_petra_head_mask, "trans_file")

                    print("Using reg_aladin transfo \
                        to pad robustpetra_skull_mask back")

                    if "petra_skull_fov" in params["skull_petra_pipe"]:
                        pad_robustpetra_skull_mask = pe.Node(
                            RegResample(inter_val="NN"),
                            name="pad_robustpetra_skull_mask")

                        main_workflow.connect(
                            skull_petra_pipe,
                            "outputnode.robustpetra_skull_mask",
                            pad_robustpetra_skull_mask, "flo_file")

                        main_workflow.connect(
                            segment_brain_pipe, "outputnode.native_T1",
                            pad_robustpetra_skull_mask, "ref_file")

                        main_workflow.connect(
                            segment_brain_pipe,
                            "outputnode.stereo_to_native_trans",
                            pad_robustpetra_skull_mask, "trans_file")

                        main_workflow.connect(
                            pad_robustpetra_skull_mask, "out_file",
                            outputnode, "native_robustpetra_skull_mask")

    if "ct" in skull_dt and "skull_ct_pipe" in params.keys():
        print("Found skull_ct_pipe")

        skull_ct_pipe = create_skull_ct_pipe(
            params=parse_key(params, "skull_ct_pipe"))

        main_workflow.connect(datasource, ('CT', get_first_elem),
                              skull_ct_pipe, 'inputnode.ct')

        main_workflow.connect(segment_brain_pipe,
                              "outputnode.native_T1",
                              skull_ct_pipe, 'inputnode.native_T1')

        main_workflow.connect(segment_brain_pipe,
                              "outputnode.native_T2",
                              skull_ct_pipe, 'inputnode.native_T2')

        main_workflow.connect(segment_brain_pipe,
                              "outputnode.stereo_padded_T1",
                              skull_ct_pipe, 'inputnode.stereo_T1')

        main_workflow.connect(
            segment_brain_pipe, "outputnode.native_to_stereo_trans",
            skull_ct_pipe, 'inputnode.native_to_stereo_trans')

    if "angio" in skull_dt and "angio_pipe" in params.keys():
        print("Found angio_pipe")

        angio_pipe = create_angio_pipe(
            params=parse_key(params, "angio_pipe"))

        main_workflow.connect(datasource, ('ANGIO', get_first_elem),
                              angio_pipe, 'inputnode.angio')

        main_workflow.connect(segment_brain_pipe,
                              "outputnode.native_T1",
                              angio_pipe, 'inputnode.native_T1')

        main_workflow.connect(segment_brain_pipe,
                              "outputnode.stereo_padded_T1",
                              angio_pipe, 'inputnode.stereo_T1')

        main_workflow.connect(segment_brain_pipe,
                              "outputnode.stereo_padded_brain_mask",
                              angio_pipe, 'inputnode.stereo_brain_mask')

        main_workflow.connect(
            segment_brain_pipe, "outputnode.native_to_stereo_trans",
            angio_pipe, 'inputnode.native_to_stereo_trans')

    if "angio" in skull_dt and "angio_quick_pipe" in params.keys():
        print("Found angio_pipe")

        if len(brain_dt):

            angio_pipe = create_quick_angio_pipe(
                params=parse_key(params, "angio_quick_pipe"))

            main_workflow.connect(datasource, ('ANGIO', get_first_elem),
                                angio_pipe, 'inputnode.angio')

            main_workflow.connect(segment_brain_pipe,
                                "outputnode.native_T1",
                                angio_pipe, 'inputnode.native_T1')

            main_workflow.connect(segment_brain_pipe,
                                "outputnode.stereo_padded_T1",
                                angio_pipe, 'inputnode.stereo_T1')

            main_workflow.connect(
                segment_brain_pipe, "outputnode.native_to_stereo_trans",
                angio_pipe, 'inputnode.native_to_stereo_trans')
        else:

            angio_pipe = create_autonomous_quick_angio_pipe(
                params=parse_key(params, "angio_quick_pipe"))

            main_workflow.connect(datasource, ('ANGIO', get_first_elem),
                                angio_pipe, 'inputnode.angio')

    if 't1' in skull_dt and "skull_t1_pipe" in params.keys():
        print("Found skull_t1_pipe")

        skull_t1_pipe = create_skull_t1_pipe(
            params=parse_key(params, "skull_t1_pipe"))

        print("Using stereo debias T1 for skull_t1_pipe ")

        if "use_debiased_t1" in params["skull_t1_pipe"].keys() and pad:
            print("Using stereo debias T1 for skull_t1_pipe ")
            main_workflow.connect(
                segment_brain_pipe, "outputnode.stereo_padded_T1",
                skull_t1_pipe, 'inputnode.stereo_T1')
        else:

            print("Using stereo native T1 for skull_t1_pipe ")
            main_workflow.connect(segment_brain_pipe,
                                  "outputnode.stereo_T1",
                                  skull_t1_pipe, 'inputnode.stereo_T1')

        if indiv_params:
            main_workflow.connect(datasource, "indiv_params",
                                  skull_t1_pipe, 'inputnode.indiv_params')

        if pad and space == "native":

            # output node
            outputnode = pe.Node(
                niu.IdentityInterface(
                    fields=["native_petra_skull_mask",
                            "native_robustpetra_skull_mask",
                            "native_petra_head_mask"]),
                name='outputnode')

            if "short_preparation_pipe" in params.keys():
                if "crop_T1" in params["short_preparation_pipe"].keys():
                    pass
                else:
                    print("Using reg_aladin transfo to pad skull_mask back")

                    pad_t1_skull_mask = pe.Node(RegResample(inter_val="NN"),
                                                name="pad_t1_skull_mask")

                    main_workflow.connect(
                        skull_t1_pipe, "outputnode.t1_skull_mask",
                        pad_t1_skull_mask, "flo_file")

                    main_workflow.connect(
                        segment_brain_pipe, "outputnode.native_T1",
                        pad_t1_skull_mask, "ref_file")

                    main_workflow.connect(
                        segment_brain_pipe,
                        "outputnode.stereo_to_native_trans",
                        pad_t1_skull_mask, "trans_file")

    if deriv:

        datasink_name = os.path.join("derivatives", wf_name)

        if "regex_subs" in params.keys():
            params_regex_subs = params["regex_subs"]
        else:
            params_regex_subs = {}

        if "subs" in params.keys():
            params_subs = params["rsubs"]
        else:
            params_subs = {}

        print(datasource.iterables)

        datasink = create_datasink(iterables=datasource.iterables,
                                   name=datasink_name,
                                   params_subs=params_subs,
                                   params_regex_subs=params_regex_subs)

        datasink.inputs.base_directory = process_dir

        if len(datasource.iterables) == 1:
            pref_deriv = "sub-%(sub)s"
            parse_str = r"sub-(?P<sub>\w*)_.*"
        elif len(datasource.iterables) > 1:
            pref_deriv = "sub-%(sub)s_ses-%(ses)s"
            parse_str = r"sub-(?P<sub>\w*)_ses-(?P<ses>\w*)_.*"

        rename_all_brain_derivatives(
            params, main_workflow, segment_brain_pipe,
            datasink, pref_deriv, parse_str, pad, ssoft,
            brain_dt)

        if "petra" in skull_dt and "skull_petra_pipe" in params.keys():
            rename_all_skull_petra_derivatives(
                params, main_workflow, segment_brain_pipe, skull_petra_pipe,
                datasink, pref_deriv, parse_str, space, pad)

            if pad:

                # rename petra_skull_mask
                rename_petra_skull_mask = pe.Node(
                    niu.Rename(), name="rename_petra_skull_mask")

                rename_petra_skull_mask.inputs.format_string = \
                    pref_deriv + "_space-native_desc-petra_skullmask"
                rename_petra_skull_mask.inputs.parse_string = parse_str
                rename_petra_skull_mask.inputs.keep_ext = True

                main_workflow.connect(
                    pad_petra_skull_mask, "out_file",
                    rename_petra_skull_mask, 'in_file')

                main_workflow.connect(
                    rename_petra_skull_mask, 'out_file',
                    datasink, '@petra_skull_mask')

                # rename petra_head_mask
                rename_petra_head_mask = pe.Node(
                    niu.Rename(), name="rename_petra_head_mask")
                rename_petra_head_mask.inputs.format_string = \
                    pref_deriv + "_space-native_desc-petra_headmask"
                rename_petra_head_mask.inputs.parse_string = parse_str
                rename_petra_head_mask.inputs.keep_ext = True

                main_workflow.connect(
                    pad_petra_head_mask, "out_file",
                    rename_petra_head_mask, 'in_file')

                main_workflow.connect(
                    rename_petra_head_mask, 'out_file',
                    datasink, '@petra_head_mask')

                if "petra_skull_fov" in params["skull_petra_pipe"]:
                    # rename robustpetra_skull_mask
                    rename_robustpetra_skull_mask = pe.Node(
                        niu.Rename(), name="rename_robustpetra_skull_mask")
                    rename_robustpetra_skull_mask.inputs.format_string = \
                        pref_deriv + \
                        "_space-native_desc-robustpetra_skullmask"

                    rename_robustpetra_skull_mask.inputs.parse_string = \
                        parse_str

                    rename_robustpetra_skull_mask.inputs.keep_ext = True

                    main_workflow.connect(
                        pad_robustpetra_skull_mask, "out_file",
                        rename_robustpetra_skull_mask, 'in_file')

                    main_workflow.connect(
                        rename_robustpetra_skull_mask, 'out_file',
                        datasink, '@robustpetra_skull_mask')

        if "t1" in skull_dt and "skull_t1_pipe" in params.keys():
            rename_all_skull_t1_derivatives(
                params, main_workflow, segment_brain_pipe, skull_t1_pipe,
                datasink, pref_deriv, parse_str, space, pad)

            if pad:

                # rename t1_skull_mask
                rename_native_t1_skull_mask = pe.Node(
                    niu.Rename(), name="rename_native_t1_skull_mask")

                rename_native_t1_skull_mask.inputs.format_string = \
                    pref_deriv + "_space-native_desc-t1_skullmask"
                rename_native_t1_skull_mask.inputs.parse_string = parse_str
                rename_native_t1_skull_mask.inputs.keep_ext = True

                main_workflow.connect(
                    pad_t1_skull_mask, "out_file",
                    rename_native_t1_skull_mask, 'in_file')

                main_workflow.connect(
                    rename_native_t1_skull_mask, 'out_file',
                    datasink, '@t1_native_skull_mask')

        if "ct" in skull_dt and "skull_ct_pipe" in params.keys():
            print("rename ct skull pipe 1")

            rename_all_skull_ct_derivatives(
                params, main_workflow, segment_brain_pipe, skull_ct_pipe,
                datasink, pref_deriv, parse_str, space, pad)

        if "angio" in skull_dt and ("angio_pipe" in params.keys()
                                    or "angio_quick_pipe" in params.keys()):

            print("rename_all_angio_derivatives")
            rename_all_angio_derivatives(params, main_workflow, angio_pipe,
                                         datasink, pref_deriv, parse_str)

    main_workflow.write_graph(graph2use="colored")
    main_workflow.config['execution'] = {'remove_unnecessary_outputs': 'false'}

    # saving real params.json
    real_params_file = op.join(process_dir, wf_name, "real_params.json")
    with open(real_params_file, 'w+') as fp:
        json.dump(params, fp)

    if deriv:
        try:
            os.makedirs(op.join(process_dir, datasink_name))
        except OSError:
            print("process_dir {} already exists".format(process_dir))

        real_params_file = op.join(process_dir,
                                   datasink_name, "real_params.json")
        with open(real_params_file, 'w+') as fp:
            json.dump(params, fp)

    if nprocs is None:
        nprocs = 4

    if "test" not in ssoft:
        if "seq" in ssoft or nprocs == 0:
            main_workflow.run()
        else:
            main_workflow.run(plugin='MultiProc',
                              plugin_args={'n_procs': nprocs})


def main():

    # Command line parser
    parser = argparse.ArgumentParser(
        description="PNH segmentation pipeline")

    parser.add_argument("-data", dest="data", type=str, required=True,
                        help="Directory containing MRI data (BIDS)")

    parser.add_argument("-out", dest="out", type=str,
                        help="Output dir", required=True)

    parser.add_argument("-soft", dest="soft", type=str,
                        help="Sofware of analysis (SPM or ANTS are defined)",
                        required=True)

    parser.add_argument("-species", dest="species", type=str,
                        help="Type of PNH to process",
                        required=False)

    parser.add_argument("-subjects", "-sub", dest="sub",
                        type=str, nargs='+', help="Subjects", required=False)

    parser.add_argument("-sessions", "-ses", dest="ses",
                        type=str, nargs='+', help="Sessions", required=False)

    parser.add_argument("-brain_datatypes", "-brain_dt", "-brain",
                        dest="brain_dt", type=str, nargs='+',
                        help="MRI Brain Datatypes (T1, T2)",
                        required=False)

    parser.add_argument("-skull_datatypes", "-skull_dt", "-skull",
                        dest="skull_dt", type=str, nargs='+',
                        help="MRI Brain Datatypes (T1, petra, CT)",
                        required=False)

    parser.add_argument("-acquisitions", "-acq", dest="acq", type=str,
                        nargs='+', default=None, help="Acquisitions")

    parser.add_argument("-records", "-rec", dest="rec", type=str, nargs='+',
                        default=None, help="Records")

    parser.add_argument("-params", dest="params_file", type=str,
                        help="Parameters json file", required=False)

    parser.add_argument("-indiv_params", "-indiv", dest="indiv_params_file",
                        type=str, help="Individual parameters json file",
                        required=False)

    parser.add_argument("-mask", dest="mask_file", type=str,
                        help="precomputed mask file", required=False)

    parser.add_argument("-template_path", dest="template_path", type=str,
                        help="specifies user-based template path",
                        required=False)

    parser.add_argument("-template_files", dest="template_files", type=str,
                        nargs="+", help="specifies user-based template files \
                            (3 or 5 are possible options)",
                        required=False)

    parser.add_argument("-nprocs", dest="nprocs", type=int,
                        help="number of processes to allocate", required=False)

    parser.add_argument("-reorient", dest="reorient", type=str,
                        help="reorient initial image", required=False)

    parser.add_argument("-deriv", dest="deriv", action='store_true',
                        help="output derivatives in BIDS orig directory",
                        required=False)

    parser.add_argument("-pad", dest="pad", action='store_true',
                        help="padding mask and seg_mask",
                        required=False)

    args = parser.parse_args()

    # main_workflow
    print("Initialising the pipeline...")
    create_main_workflow(
        data_dir=args.data,
        soft=args.soft,
        process_dir=args.out,
        species=args.species,
        subjects=args.sub,
        sessions=args.ses,
        brain_dt=args.brain_dt,
        skull_dt=args.skull_dt,
        acquisitions=args.acq,
        reconstructions=args.rec,
        params_file=args.params_file,
        indiv_params_file=args.indiv_params_file,
        mask_file=args.mask_file,
        template_path=args.template_path,
        template_files=args.template_files,
        nprocs=args.nprocs,
        reorient=args.reorient,
        deriv=args.deriv,
        pad=args.pad)


if __name__ == '__main__':
    main()
