??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
v
layer0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayer0/kernel
o
!layer0/kernel/Read/ReadVariableOpReadVariableOplayer0/kernel*
_output_shapes

:*
dtype0
n
layer0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer0/bias
g
layer0/bias/Read/ReadVariableOpReadVariableOplayer0/bias*
_output_shapes
:*
dtype0
v
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namelayer1/kernel
o
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes

:
*
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
:
*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:
*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
? 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*
:
%iter
	&decay
'learning_rate
(momentum*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
,
)0
*1
+2
,3
-4
.5* 
?
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

4serving_default* 
]W
VARIABLE_VALUElayer0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

)0
*1* 
?
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUElayer1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

+0
,1* 
?
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

-0
.1* 
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 

0
1
2*

D0
E1*
* 
* 
* 
* 
* 
* 

)0
*1* 
* 
* 
* 
* 

+0
,1* 
* 
* 
* 
* 

-0
.1* 
* 
8
	Ftotal
	Gcount
H	variables
I	keras_api*
H
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

H	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

J0
K1*

M	variables*

serving_default_layer0_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_layer0_inputlayer0/kernellayer0/biaslayer1/kernellayer1/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *,
f'R%
#__inference_signature_wrapper_12535
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer0/kernel/Read/ReadVariableOplayer0/bias/Read/ReadVariableOp!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *'
f"R 
__inference__traced_save_12797
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer0/kernellayer0/biaslayer1/kernellayer1/biasoutput/kerneloutput/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__traced_restore_12849??
?I
?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12516

inputs7
%layer0_matmul_readvariableop_resource:4
&layer0_biasadd_readvariableop_resource:7
%layer1_matmul_readvariableop_resource:
4
&layer1_biasadd_readvariableop_resource:
7
%output_matmul_readvariableop_resource:
4
&output_biasadd_readvariableop_resource:
identity??layer0/BiasAdd/ReadVariableOp?layer0/MatMul/ReadVariableOp?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/MatMul/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0w
layer0/MatMulMatMulinputs$layer0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
layer0/BiasAdd/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/BiasAddBiasAddlayer0/MatMul:product:0%layer0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
layer0/ReluRelulayer0/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
layer1/MatMulMatMullayer0/Relu:activations:0$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
^
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
output/MatMulMatMullayer1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer0/kernel/Regularizer/SumSum$layer0/kernel/Regularizer/Square:y:0(layer0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer0/bias/Regularizer/SumSum"layer0/bias/Regularizer/Square:y:0&layer0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
g
layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer1/bias/Regularizer/SumSum"layer1/bias/Regularizer/Square:y:0&layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
output/kernel/Regularizer/SumSum$output/kernel/Regularizer/Square:y:0(output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
output/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
output/bias/Regularizer/SumSum"output/bias/Regularizer/Square:y:0&output/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/BiasAdd/ReadVariableOp^layer0/MatMul/ReadVariableOp.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2>
layer0/BiasAdd/ReadVariableOplayer0/BiasAdd/ReadVariableOp2<
layer0/MatMul/ReadVariableOplayer0/MatMul/ReadVariableOp2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_output_layer_call_and_return_conditional_losses_12666

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
output/kernel/Regularizer/SumSum$output/kernel/Regularizer/Square:y:0(output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
output/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
output/bias/Regularizer/SumSum"output/bias/Regularizer/Square:y:0&output/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
A__inference_layer1_layer_call_and_return_conditional_losses_11990

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
g
layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer1/bias/Regularizer/SumSum"layer1/bias/Regularizer/Square:y:0&layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12322
layer0_input
layer0_12270:
layer0_12272:
layer1_12275:

layer1_12277:

output_12280:

output_12282:
identity??layer0/StatefulPartitionedCall?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?output/StatefulPartitionedCall?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/StatefulPartitionedCallStatefulPartitionedCalllayer0_inputlayer0_12270layer0_12272*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer0_layer_call_and_return_conditional_losses_11961?
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_12275layer1_12277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_11990?
output/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0output_12280output_12282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_12018|
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer0_12270*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer0/kernel/Regularizer/SumSum$layer0/kernel/Regularizer/Square:y:0(layer0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer0_12272*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer0/bias/Regularizer/SumSum"layer0/bias/Regularizer/Square:y:0&layer0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_12275*
_output_shapes

:
*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer1_12277*
_output_shapes
:
*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
g
layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer1/bias/Regularizer/SumSum"layer1/bias/Regularizer/Square:y:0&layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_12280*
_output_shapes

:
*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
output/kernel/Regularizer/SumSum$output/kernel/Regularizer/Square:y:0(output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpoutput_12282*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
output/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
output/bias/Regularizer/SumSum"output/bias/Regularizer/Square:y:0&output/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/StatefulPartitionedCall.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?
?
#__inference_signature_wrapper_12535
layer0_input
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *)
f$R"
 __inference__wrapped_model_11931o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?
?
&__inference_layer1_layer_call_fn_12600

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_11990o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
 __inference__wrapped_model_11931
layer0_input@
.layer_nn_layer0_matmul_readvariableop_resource:=
/layer_nn_layer0_biasadd_readvariableop_resource:@
.layer_nn_layer1_matmul_readvariableop_resource:
=
/layer_nn_layer1_biasadd_readvariableop_resource:
@
.layer_nn_output_matmul_readvariableop_resource:
=
/layer_nn_output_biasadd_readvariableop_resource:
identity??(3_layer_NN/layer0/BiasAdd/ReadVariableOp?'3_layer_NN/layer0/MatMul/ReadVariableOp?(3_layer_NN/layer1/BiasAdd/ReadVariableOp?'3_layer_NN/layer1/MatMul/ReadVariableOp?(3_layer_NN/output/BiasAdd/ReadVariableOp?'3_layer_NN/output/MatMul/ReadVariableOp?
'3_layer_NN/layer0/MatMul/ReadVariableOpReadVariableOp.layer_nn_layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
3_layer_NN/layer0/MatMulMatMullayer0_input/3_layer_NN/layer0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(3_layer_NN/layer0/BiasAdd/ReadVariableOpReadVariableOp/layer_nn_layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
3_layer_NN/layer0/BiasAddBiasAdd"3_layer_NN/layer0/MatMul:product:003_layer_NN/layer0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
3_layer_NN/layer0/ReluRelu"3_layer_NN/layer0/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'3_layer_NN/layer1/MatMul/ReadVariableOpReadVariableOp.layer_nn_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
3_layer_NN/layer1/MatMulMatMul$3_layer_NN/layer0/Relu:activations:0/3_layer_NN/layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
(3_layer_NN/layer1/BiasAdd/ReadVariableOpReadVariableOp/layer_nn_layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
3_layer_NN/layer1/BiasAddBiasAdd"3_layer_NN/layer1/MatMul:product:003_layer_NN/layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
t
3_layer_NN/layer1/ReluRelu"3_layer_NN/layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
'3_layer_NN/output/MatMul/ReadVariableOpReadVariableOp.layer_nn_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
3_layer_NN/output/MatMulMatMul$3_layer_NN/layer1/Relu:activations:0/3_layer_NN/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(3_layer_NN/output/BiasAdd/ReadVariableOpReadVariableOp/layer_nn_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
3_layer_NN/output/BiasAddBiasAdd"3_layer_NN/output/MatMul:product:003_layer_NN/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"3_layer_NN/output/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^3_layer_NN/layer0/BiasAdd/ReadVariableOp(^3_layer_NN/layer0/MatMul/ReadVariableOp)^3_layer_NN/layer1/BiasAdd/ReadVariableOp(^3_layer_NN/layer1/MatMul/ReadVariableOp)^3_layer_NN/output/BiasAdd/ReadVariableOp(^3_layer_NN/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2T
(3_layer_NN/layer0/BiasAdd/ReadVariableOp(3_layer_NN/layer0/BiasAdd/ReadVariableOp2R
'3_layer_NN/layer0/MatMul/ReadVariableOp'3_layer_NN/layer0/MatMul/ReadVariableOp2T
(3_layer_NN/layer1/BiasAdd/ReadVariableOp(3_layer_NN/layer1/BiasAdd/ReadVariableOp2R
'3_layer_NN/layer1/MatMul/ReadVariableOp'3_layer_NN/layer1/MatMul/ReadVariableOp2T
(3_layer_NN/output/BiasAdd/ReadVariableOp(3_layer_NN/output/BiasAdd/ReadVariableOp2R
'3_layer_NN/output/MatMul/ReadVariableOp'3_layer_NN/output/MatMul/ReadVariableOp:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?

?
__inference_loss_fn_1_12688D
6layer0_bias_regularizer_square_readvariableop_resource:
identity??-layer0/bias/Regularizer/Square/ReadVariableOp?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOp6layer0_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer0/bias/Regularizer/SumSum"layer0/bias/Regularizer/Square:y:0&layer0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentitylayer0/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: v
NoOpNoOp.^layer0/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp
?8
?
!__inference__traced_restore_12849
file_prefix0
assignvariableop_layer0_kernel:,
assignvariableop_1_layer0_bias:2
 assignvariableop_2_layer1_kernel:
,
assignvariableop_3_layer1_bias:
2
 assignvariableop_4_output_kernel:
,
assignvariableop_5_output_bias:%
assignvariableop_6_sgd_iter:	 &
assignvariableop_7_sgd_decay: .
$assignvariableop_8_sgd_learning_rate: )
assignvariableop_9_sgd_momentum: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: 
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_layer0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12267
layer0_input
layer0_12215:
layer0_12217:
layer1_12220:

layer1_12222:

output_12225:

output_12227:
identity??layer0/StatefulPartitionedCall?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?output/StatefulPartitionedCall?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/StatefulPartitionedCallStatefulPartitionedCalllayer0_inputlayer0_12215layer0_12217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer0_layer_call_and_return_conditional_losses_11961?
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_12220layer1_12222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_11990?
output/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0output_12225output_12227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_12018|
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer0_12215*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer0/kernel/Regularizer/SumSum$layer0/kernel/Regularizer/Square:y:0(layer0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer0_12217*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer0/bias/Regularizer/SumSum"layer0/bias/Regularizer/Square:y:0&layer0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_12220*
_output_shapes

:
*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer1_12222*
_output_shapes
:
*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
g
layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer1/bias/Regularizer/SumSum"layer1/bias/Regularizer/Square:y:0&layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_12225*
_output_shapes

:
*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
output/kernel/Regularizer/SumSum$output/kernel/Regularizer/Square:y:0(output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpoutput_12227*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
output/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
output/bias/Regularizer/SumSum"output/bias/Regularizer/Square:y:0&output/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/StatefulPartitionedCall.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?
?
*__inference_3_layer_NN_layer_call_fn_12379

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?I
?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12456

inputs7
%layer0_matmul_readvariableop_resource:4
&layer0_biasadd_readvariableop_resource:7
%layer1_matmul_readvariableop_resource:
4
&layer1_biasadd_readvariableop_resource:
7
%output_matmul_readvariableop_resource:
4
&output_biasadd_readvariableop_resource:
identity??layer0/BiasAdd/ReadVariableOp?layer0/MatMul/ReadVariableOp?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/MatMul/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0w
layer0/MatMulMatMulinputs$layer0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
layer0/BiasAdd/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/BiasAddBiasAddlayer0/MatMul:product:0%layer0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
layer0/ReluRelulayer0/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
layer1/MatMulMatMullayer0/Relu:activations:0$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
^
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
output/MatMulMatMullayer1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer0/kernel/Regularizer/SumSum$layer0/kernel/Regularizer/Square:y:0(layer0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer0/bias/Regularizer/SumSum"layer0/bias/Regularizer/Square:y:0&layer0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
g
layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer1/bias/Regularizer/SumSum"layer1/bias/Regularizer/Square:y:0&layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
output/kernel/Regularizer/SumSum$output/kernel/Regularizer/Square:y:0(output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
output/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
output/bias/Regularizer/SumSum"output/bias/Regularizer/Square:y:0&output/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/BiasAdd/ReadVariableOp^layer0/MatMul/ReadVariableOp.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2>
layer0/BiasAdd/ReadVariableOplayer0/BiasAdd/ReadVariableOp2<
layer0/MatMul/ReadVariableOplayer0/MatMul/ReadVariableOp2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_3_layer_NN_layer_call_fn_12396

inputs
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_12699J
8layer1_kernel_regularizer_square_readvariableop_resource:

identity??/layer1/kernel/Regularizer/Square/ReadVariableOp?
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8layer1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:
*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!layer1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^layer1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp
?>
?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12061

inputs
layer0_11962:
layer0_11964:
layer1_11991:

layer1_11993:

output_12019:

output_12021:
identity??layer0/StatefulPartitionedCall?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?output/StatefulPartitionedCall?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/StatefulPartitionedCallStatefulPartitionedCallinputslayer0_11962layer0_11964*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer0_layer_call_and_return_conditional_losses_11961?
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_11991layer1_11993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_11990?
output/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0output_12019output_12021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_12018|
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer0_11962*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer0/kernel/Regularizer/SumSum$layer0/kernel/Regularizer/Square:y:0(layer0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer0_11964*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer0/bias/Regularizer/SumSum"layer0/bias/Regularizer/Square:y:0&layer0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_11991*
_output_shapes

:
*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer1_11993*
_output_shapes
:
*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
g
layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer1/bias/Regularizer/SumSum"layer1/bias/Regularizer/Square:y:0&layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_12019*
_output_shapes

:
*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
output/kernel/Regularizer/SumSum$output/kernel/Regularizer/Square:y:0(output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpoutput_12021*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
output/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
output/bias/Regularizer/SumSum"output/bias/Regularizer/Square:y:0&output/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/StatefulPartitionedCall.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_output_layer_call_and_return_conditional_losses_12018

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
output/kernel/Regularizer/SumSum$output/kernel/Regularizer/Square:y:0(output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
output/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
output/bias/Regularizer/SumSum"output/bias/Regularizer/Square:y:0&output/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?$
?
__inference__traced_save_12797
file_prefix,
(savev2_layer0_kernel_read_readvariableop*
&savev2_layer0_bias_read_readvariableop,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer0_kernel_read_readvariableop&savev2_layer0_bias_read_readvariableop(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*W
_input_shapesF
D: :::
:
:
:: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_loss_fn_4_12721J
8output_kernel_regularizer_square_readvariableop_resource:

identity??/output/kernel/Regularizer/Square/ReadVariableOp?
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8output_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:
*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
output/kernel/Regularizer/SumSum$output/kernel/Regularizer/Square:y:0(output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!output/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp
?

?
__inference_loss_fn_5_12732D
6output_bias_regularizer_square_readvariableop_resource:
identity??-output/bias/Regularizer/Square/ReadVariableOp?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOp6output_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
output/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
output/bias/Regularizer/SumSum"output/bias/Regularizer/Square:y:0&output/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentityoutput/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: v
NoOpNoOp.^output/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp
?

?
__inference_loss_fn_3_12710D
6layer1_bias_regularizer_square_readvariableop_resource:

identity??-layer1/bias/Regularizer/Square/ReadVariableOp?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOp6layer1_bias_regularizer_square_readvariableop_resource*
_output_shapes
:
*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
g
layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer1/bias/Regularizer/SumSum"layer1/bias/Regularizer/Square:y:0&layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentitylayer1/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: v
NoOpNoOp.^layer1/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp
?	
?
*__inference_3_layer_NN_layer_call_fn_12076
layer0_input
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?
?
A__inference_layer0_layer_call_and_return_conditional_losses_12579

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer0/kernel/Regularizer/SumSum$layer0/kernel/Regularizer/Square:y:0(layer0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer0/bias/Regularizer/SumSum"layer0/bias/Regularizer/Square:y:0&layer0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_12677J
8layer0_kernel_regularizer_square_readvariableop_resource:
identity??/layer0/kernel/Regularizer/Square/ReadVariableOp?
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8layer0_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer0/kernel/Regularizer/SumSum$layer0/kernel/Regularizer/Square:y:0(layer0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!layer0/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^layer0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp
?
?
A__inference_layer1_layer_call_and_return_conditional_losses_12623

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
g
layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer1/bias/Regularizer/SumSum"layer1/bias/Regularizer/Square:y:0&layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_output_layer_call_fn_12644

inputs
unknown:

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_12018o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
A__inference_layer0_layer_call_and_return_conditional_losses_11961

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer0/kernel/Regularizer/SumSum$layer0/kernel/Regularizer/Square:y:0(layer0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer0/bias/Regularizer/SumSum"layer0/bias/Regularizer/Square:y:0&layer0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_layer0_layer_call_fn_12556

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer0_layer_call_and_return_conditional_losses_11961o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_3_layer_NN_layer_call_fn_12212
layer0_input
unknown:
	unknown_0:
	unknown_1:

	unknown_2:

	unknown_3:

	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?>
?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12180

inputs
layer0_12128:
layer0_12130:
layer1_12133:

layer1_12135:

output_12138:

output_12140:
identity??layer0/StatefulPartitionedCall?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?output/StatefulPartitionedCall?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/StatefulPartitionedCallStatefulPartitionedCallinputslayer0_12128layer0_12130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer0_layer_call_and_return_conditional_losses_11961?
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_12133layer1_12135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_11990?
output/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0output_12138output_12140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_12018|
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer0_12128*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer0/kernel/Regularizer/SumSum$layer0/kernel/Regularizer/Square:y:0(layer0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer0_12130*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer0/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer0/bias/Regularizer/SumSum"layer0/bias/Regularizer/Square:y:0&layer0/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer0/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_12133*
_output_shapes

:
*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer1_12135*
_output_shapes
:
*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:
g
layer1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer1/bias/Regularizer/SumSum"layer1/bias/Regularizer/Square:y:0&layer1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_12138*
_output_shapes

:
*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:
p
output/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
output/kernel/Regularizer/SumSum$output/kernel/Regularizer/Square:y:0(output/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
output/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpoutput_12140*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
output/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
output/bias/Regularizer/SumSum"output/bias/Regularizer/Square:y:0&output/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
output/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/StatefulPartitionedCall.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
layer0_input5
serving_default_layer0_input:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?V
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
I
%iter
	&decay
'learning_rate
(momentum"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
)0
*1
+2
,3
-4
.5"
trackable_list_wrapper
?
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_3_layer_NN_layer_call_fn_12076
*__inference_3_layer_NN_layer_call_fn_12379
*__inference_3_layer_NN_layer_call_fn_12396
*__inference_3_layer_NN_layer_call_fn_12212?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12456
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12516
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12267
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12322?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_11931layer0_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
4serving_default"
signature_map
:2layer0/kernel
:2layer0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer0_layer_call_fn_12556?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer0_layer_call_and_return_conditional_losses_12579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:
2layer1/kernel
:
2layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_layer1_layer_call_fn_12600?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer1_layer_call_and_return_conditional_losses_12623?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:
2output/kernel
:2output/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_output_layer_call_fn_12644?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_output_layer_call_and_return_conditional_losses_12666?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
?2?
__inference_loss_fn_0_12677?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_12688?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_12699?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_12710?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_12721?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_12732?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_signature_wrapper_12535layer0_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ftotal
	Gcount
H	variables
I	keras_api"
_tf_keras_metric
^
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
F0
G1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
-
M	variables"
_generic_user_object?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12267n=?:
3?0
&?#
layer0_input?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12322n=?:
3?0
&?#
layer0_input?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12456h7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_3_layer_NN_layer_call_and_return_conditional_losses_12516h7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_3_layer_NN_layer_call_fn_12076a=?:
3?0
&?#
layer0_input?????????
p 

 
? "???????????
*__inference_3_layer_NN_layer_call_fn_12212a=?:
3?0
&?#
layer0_input?????????
p

 
? "???????????
*__inference_3_layer_NN_layer_call_fn_12379[7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
*__inference_3_layer_NN_layer_call_fn_12396[7?4
-?*
 ?
inputs?????????
p

 
? "???????????
 __inference__wrapped_model_11931p5?2
+?(
&?#
layer0_input?????????
? "/?,
*
output ?
output??????????
A__inference_layer0_layer_call_and_return_conditional_losses_12579\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_layer0_layer_call_fn_12556O/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_layer1_layer_call_and_return_conditional_losses_12623\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? y
&__inference_layer1_layer_call_fn_12600O/?,
%?"
 ?
inputs?????????
? "??????????
:
__inference_loss_fn_0_12677?

? 
? "? :
__inference_loss_fn_1_12688?

? 
? "? :
__inference_loss_fn_2_12699?

? 
? "? :
__inference_loss_fn_3_12710?

? 
? "? :
__inference_loss_fn_4_12721?

? 
? "? :
__inference_loss_fn_5_12732?

? 
? "? ?
A__inference_output_layer_call_and_return_conditional_losses_12666\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? y
&__inference_output_layer_call_fn_12644O/?,
%?"
 ?
inputs?????????

? "???????????
#__inference_signature_wrapper_12535?E?B
? 
;?8
6
layer0_input&?#
layer0_input?????????"/?,
*
output ?
output?????????