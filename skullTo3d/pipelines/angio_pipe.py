"""
    Gather all full pipelines

"""
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe


from nipype.interfaces.fsl.maths import (
    UnaryMaths, Threshold, ApplyMask)

from nipype.interfaces.fsl.preprocess import FAST


from nipype.interfaces.niftyreg.regutils import RegResample
from nipype.interfaces.niftyreg.reg import RegAladin

from macapype.utils.utils_nodes import NodeParams

from macapype.nodes.denoise import DenoiseImage

from macapype.nodes.surface import (keep_gcc)

from macapype.utils.misc import parse_key, get_elem

###############################################################################
# #################### ANGIO  ######################
###############################################################################


def create_angio_pipe(name="angio_pipe", params={}):

    # Creating pipeline
    angio_pipe = pe.Workflow(name=name)

    # Creating input node
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['angio', 'stereo_T1', 'native_T1',
                                      'stereo_brain_mask',
                                      'native_to_stereo_trans',
                                      'indiv_params']),
        name='inputnode'
    )

    # creating outputnode #######
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["stereo_angio", "stereo_brain_angio",
                    "stereo_angio_mask"]),
        name='outputnode')

    # align_angio_on_T1
    align_angio_on_T1 = pe.Node(interface=RegAladin(),
                                name="align_angio_on_T1")

    angio_pipe.connect(inputnode, 'angio',
                       align_angio_on_T1, "flo_file")

    angio_pipe.connect(inputnode, "native_T1",
                       align_angio_on_T1, "ref_file")

    # align_angio_on_stereo_T1
    align_angio_on_stereo_T1 = pe.Node(
        interface=RegResample(pad_val=0.0),
        name="align_angio_on_stereo_T1")

    angio_pipe.connect(align_angio_on_T1, 'res_file',
                       align_angio_on_stereo_T1, "flo_file")

    angio_pipe.connect(inputnode, 'native_to_stereo_trans',
                       align_angio_on_stereo_T1, "trans_file")

    angio_pipe.connect(inputnode, "stereo_T1",
                       align_angio_on_stereo_T1, "ref_file")

    # outputs
    angio_pipe.connect(align_angio_on_stereo_T1, "out_file",
                       outputnode, 'stereo_angio')

    # angio_denoise
    angio_denoise = NodeParams(interface=DenoiseImage(),
                               params=parse_key(params, "angio_denoise"),
                               name="angio_denoise")
    angio_pipe.connect(
        align_angio_on_stereo_T1, "out_file",
        angio_denoise, 'input_image')

    angio_pipe.connect(
        inputnode, "stereo_brain_mask",
        angio_denoise, 'mask_image')

    # outputs
    angio_pipe.connect(angio_denoise, 'output_image',
                       outputnode, 'stereo_brain_angio')

    # angio_auto_thresh
    if "angio_mask_thr" in params.keys():

        print("*** angio_mask_thr ***")

        # angio_mask_thr ####### [okey][json]
        angio_mask_thr = NodeParams(
            interface=Threshold(),
            params=parse_key(params, "angio_mask_thr"),
            name="angio_mask_thr")

        angio_pipe.connect(
            inputnode, ("indiv_params", parse_key, "angio_mask_thr"),
            angio_mask_thr, "indiv_params")

        angio_pipe.connect(angio_denoise, 'output_image',
                           angio_mask_thr, "in_file")

    else:

        print("*** angio_fast ***")

        angio_fast = NodeParams(
            interface=FAST(),
            params=parse_key(params, "angio_fast"),
            name="angio_fast")

        angio_pipe.connect(
            angio_denoise, 'output_image',
            angio_fast, "in_files")

        angio_pipe.connect(
            inputnode, ('indiv_params', parse_key, "angio_fast"),
            angio_fast, "indiv_params")

    # angio_mask_binary
    angio_mask_binary = pe.Node(interface=UnaryMaths(),
                                name="angio_mask_binary")

    angio_mask_binary.inputs.operation = 'bin'
    angio_mask_binary.inputs.output_type = 'NIFTI_GZ'

    if "angio_mask_thr" in params.keys():

        angio_pipe.connect(
            angio_mask_thr, "out_file",
            angio_mask_binary, "in_file")
    else:

        angio_pipe.connect(
            angio_fast, ("partial_volume_files", get_elem, 1),
            angio_mask_binary, "in_file")

    # angio_gcc ####### [okey]
    angio_gcc = pe.Node(
        interface=niu.Function(
            input_names=["nii_file"],
            output_names=["gcc_nii_file"],
            function=keep_gcc),
        name="angio_gcc")

    angio_pipe.connect(
        angio_mask_binary, "out_file",
        angio_gcc, "nii_file")

    # angio_bmasked ####### [okey]
    angio_bmasked = pe.Node(
        ApplyMask(),
        name="angio_bmasked")

    angio_pipe.connect(
        angio_gcc, "gcc_nii_file",
        angio_bmasked, "in_file")

    angio_pipe.connect(
        inputnode, "stereo_brain_mask",
        angio_bmasked, "mask_file")

    # outputnode
    angio_pipe.connect(angio_bmasked, "out_file",
                       outputnode, 'stereo_angio_mask')

    return angio_pipe

def create_autonomous_quick_angio_pipe(name="quick_angio_pipe", params={}):

    # Creating pipeline
    angio_pipe = pe.Workflow(name=name)

    # Creating input node
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['angio',
                                      'indiv_params']),
        name='inputnode'
    )

    # creating outputnode #######
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["stereo_angio", "stereo_angio_mask"]),
        name='outputnode')

    # angio_auto_thresh
    if "angio_mask_thr" in params.keys():

        print("*** angio_mask_thr ***")

        # angio_mask_thr ####### [okey][json]
        angio_mask_thr = NodeParams(
            interface=Threshold(),
            params=parse_key(params, "angio_mask_thr"),
            name="angio_mask_thr")

        angio_pipe.connect(
            inputnode, ("indiv_params", parse_key, "angio_mask_thr"),
            angio_mask_thr, "indiv_params")

        angio_pipe.connect(inputnode, 'angio',
                           angio_mask_thr, "in_file")

    else:

        print("*** angio_auto_mask ***")

        angio_fast = NodeParams(
            interface=FAST(),
            params=parse_key(params, "angio_fast"),
            name="angio_fast")

        angio_pipe.connect(inputnode, 'angio',
                           angio_fast, "in_files")

        angio_pipe.connect(
            inputnode, ('indiv_params', parse_key, "angio_fast"),
            angio_fast, "indiv_params")

    # angio_mask_binary
    angio_mask_binary = pe.Node(interface=UnaryMaths(),
                                name="angio_mask_binary")

    angio_mask_binary.inputs.operation = 'bin'
    angio_mask_binary.inputs.output_type = 'NIFTI_GZ'

    if "angio_mask_thr" in params.keys():

        angio_pipe.connect(
            angio_mask_thr, "out_file",
            angio_mask_binary, "in_file")

    else:
        angio_pipe.connect(
            angio_fast, ("partial_volume_files", get_elem, 0),
            angio_mask_binary, "in_file")

    # angio_gcc ####### [okey]
    angio_gcc = pe.Node(
        interface=niu.Function(
            input_names=["nii_file"],
            output_names=["gcc_nii_file"],
            function=keep_gcc),
        name="angio_gcc")

    angio_pipe.connect(
        angio_mask_binary, "out_file",
        angio_gcc, "nii_file")

    angio_pipe.connect(angio_gcc, 'gcc_nii_file',
                       outputnode, 'stereo_angio_mask')

    return angio_pipe


def create_quick_angio_pipe(name="quick_angio_pipe", params={}):

    # Creating pipeline
    angio_pipe = pe.Workflow(name=name)

    # Creating input node
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['angio', 'stereo_T1', 'native_T1',
                                      'native_to_stereo_trans',
                                      'indiv_params']),
        name='inputnode'
    )

    # creating outputnode #######
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["stereo_angio", "stereo_angio_mask"]),
        name='outputnode')

    # align_angio_on_T1
    align_angio_on_T1 = pe.Node(interface=RegAladin(),
                                name="align_angio_on_T1")

    angio_pipe.connect(inputnode, 'angio',
                       align_angio_on_T1, "flo_file")

    angio_pipe.connect(inputnode, "native_T1",
                       align_angio_on_T1, "ref_file")

    # align_angio_on_stereo_T1
    align_angio_on_stereo_T1 = pe.Node(
        interface=RegResample(pad_val=0.0),
        name="align_angio_on_stereo_T1")

    angio_pipe.connect(align_angio_on_T1, 'res_file',
                       align_angio_on_stereo_T1, "flo_file")

    angio_pipe.connect(inputnode, 'native_to_stereo_trans',
                       align_angio_on_stereo_T1, "trans_file")

    angio_pipe.connect(inputnode, "stereo_T1",
                       align_angio_on_stereo_T1, "ref_file")

    # outputs
    angio_pipe.connect(align_angio_on_stereo_T1, "out_file",
                       outputnode, 'stereo_angio')

    # angio_auto_thresh
    if "angio_mask_thr" in params.keys():

        print("*** angio_mask_thr ***")

        # angio_mask_thr ####### [okey][json]
        angio_mask_thr = NodeParams(
            interface=Threshold(),
            params=parse_key(params, "angio_mask_thr"),
            name="angio_mask_thr")

        angio_pipe.connect(
            inputnode, ("indiv_params", parse_key, "angio_mask_thr"),
            angio_mask_thr, "indiv_params")

        angio_pipe.connect(align_angio_on_stereo_T1, "out_file",
                           angio_mask_thr, "in_file")

    else:

        print("*** angio_auto_mask ***")

        angio_fast = NodeParams(
            interface=FAST(),
            params=parse_key(params, "angio_fast"),
            name="angio_fast")

        angio_pipe.connect(
            align_angio_on_stereo_T1, "out_file",
            angio_fast, "in_files")

        angio_pipe.connect(
            inputnode, ('indiv_params', parse_key, "angio_fast"),
            angio_fast, "indiv_params")

    # angio_mask_binary
    angio_mask_binary = pe.Node(interface=UnaryMaths(),
                                name="angio_mask_binary")

    angio_mask_binary.inputs.operation = 'bin'
    angio_mask_binary.inputs.output_type = 'NIFTI_GZ'

    if "angio_mask_thr" in params.keys():

        angio_pipe.connect(
            angio_mask_thr, "out_file",
            angio_mask_binary, "in_file")

    else:
        angio_pipe.connect(
            angio_fast, ("partial_volume_files", get_elem, 0),
            angio_mask_binary, "in_file")

    # angio_gcc ####### [okey]
    angio_gcc = pe.Node(
        interface=niu.Function(
            input_names=["nii_file"],
            output_names=["gcc_nii_file"],
            function=keep_gcc),
        name="angio_gcc")

    angio_pipe.connect(
        angio_mask_binary, "out_file",
        angio_gcc, "nii_file")

    angio_pipe.connect(angio_gcc, 'gcc_nii_file',
                       outputnode, 'stereo_angio_mask')

    return angio_pipe
