??

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
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??	
v
layer0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayer0/kernel
o
!layer0/kernel/Read/ReadVariableOpReadVariableOplayer0/kernel*
_output_shapes

:*
dtype0
n
layer0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer0/bias
g
layer0/bias/Read/ReadVariableOpReadVariableOplayer0/bias*
_output_shapes
:*
dtype0
v
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayer1/kernel
o
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes

:*
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
:*
dtype0
v
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayer2/kernel
o
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes

:*
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
?
Adam/layer0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer0/kernel/m
}
(Adam/layer0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer0/kernel/m*
_output_shapes

:*
dtype0
|
Adam/layer0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer0/bias/m
u
&Adam/layer0/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer0/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer1/kernel/m
}
(Adam/layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/m*
_output_shapes

:*
dtype0
|
Adam/layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer1/bias/m
u
&Adam/layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer2/kernel/m
}
(Adam/layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/m*
_output_shapes

:*
dtype0
|
Adam/layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/m
u
&Adam/layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/m*
_output_shapes
:*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer0/kernel/v
}
(Adam/layer0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer0/kernel/v*
_output_shapes

:*
dtype0
|
Adam/layer0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer0/bias/v
u
&Adam/layer0/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer0/bias/v*
_output_shapes
:*
dtype0
?
Adam/layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer1/kernel/v
}
(Adam/layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/v*
_output_shapes

:*
dtype0
|
Adam/layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer1/bias/v
u
&Adam/layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/v*
_output_shapes
:*
dtype0
?
Adam/layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer2/kernel/v
}
(Adam/layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/v*
_output_shapes

:*
dtype0
|
Adam/layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/v
u
&Adam/layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/v*
_output_shapes
:*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?8
value?8B?8 B?8
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
?

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*
?
.iter

/beta_1

0beta_2
	1decay
2learning_ratem`mambmcmdme&mf'mgvhvivjvkvlvm&vn'vo*
<
0
1
2
3
4
5
&6
'7*
<
0
1
2
3
4
5
&6
'7*
:
30
41
52
63
74
85
96
:7* 
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

@serving_default* 
]W
VARIABLE_VALUElayer0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

30
41* 
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUElayer1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

50
61* 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUElayer2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElayer2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

70
81* 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*

90
:1* 
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
 
0
1
2
3*

U0
V1*
* 
* 
* 
* 
* 
* 

30
41* 
* 
* 
* 
* 

50
61* 
* 
* 
* 
* 

70
81* 
* 
* 
* 
* 

90
:1* 
* 
8
	Wtotal
	Xcount
Y	variables
Z	keras_api*
H
	[total
	\count
]
_fn_kwargs
^	variables
_	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

W0
X1*

Y	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

[0
\1*

^	variables*
?z
VARIABLE_VALUEAdam/layer0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/layer0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/layer1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/layer1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/layer2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/layer2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/layer0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/layer0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/layer1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/layer1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/layer2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/layer2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_layer0_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_layer0_inputlayer0/kernellayer0/biaslayer1/kernellayer1/biaslayer2/kernellayer2/biasoutput/kerneloutput/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_237633
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer0/kernel/Read/ReadVariableOplayer0/bias/Read/ReadVariableOp!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/layer0/kernel/m/Read/ReadVariableOp&Adam/layer0/bias/m/Read/ReadVariableOp(Adam/layer1/kernel/m/Read/ReadVariableOp&Adam/layer1/bias/m/Read/ReadVariableOp(Adam/layer2/kernel/m/Read/ReadVariableOp&Adam/layer2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp(Adam/layer0/kernel/v/Read/ReadVariableOp&Adam/layer0/bias/v/Read/ReadVariableOp(Adam/layer1/kernel/v/Read/ReadVariableOp&Adam/layer1/bias/v/Read/ReadVariableOp(Adam/layer2/kernel/v/Read/ReadVariableOp&Adam/layer2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_238018
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer0/kernellayer0/biaslayer1/kernellayer1/biaslayer2/kernellayer2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/layer0/kernel/mAdam/layer0/bias/mAdam/layer1/kernel/mAdam/layer1/bias/mAdam/layer2/kernel/mAdam/layer2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/layer0/kernel/vAdam/layer0/bias/vAdam/layer1/kernel/vAdam/layer1/bias/vAdam/layer2/kernel/vAdam/layer2/bias/vAdam/output/kernel/vAdam/output/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_238127??
?
?
B__inference_layer1_layer_call_and_return_conditional_losses_236906

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_237841J
8layer1_kernel_regularizer_square_readvariableop_resource:
identity??/layer1/kernel/Regularizer/Square/ReadVariableOp?
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8layer1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
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
?
?
__inference_loss_fn_4_237863J
8layer2_kernel_regularizer_square_readvariableop_resource:
identity??/layer2/kernel/Regularizer/Square/ReadVariableOp?
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8layer2_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!layer2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^layer2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_6_237885J
8output_kernel_regularizer_square_readvariableop_resource:
identity??/output/kernel/Regularizer/Square/ReadVariableOp?
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8output_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
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
?R
?
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237284
layer0_input
layer0_237215:
layer0_237217:
layer1_237220:
layer1_237222:
layer2_237225:
layer2_237227:
output_237230:
output_237232:
identity??layer0/StatefulPartitionedCall?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?layer2/StatefulPartitionedCall?-layer2/bias/Regularizer/Square/ReadVariableOp?/layer2/kernel/Regularizer/Square/ReadVariableOp?output/StatefulPartitionedCall?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/StatefulPartitionedCallStatefulPartitionedCalllayer0_inputlayer0_237215layer0_237217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_236877?
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_237220layer1_237222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_236906?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_237225layer2_237227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_236935?
output/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0output_237230output_237232*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_236963}
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer0_237215*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer0_237217*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_237220*
_output_shapes

:*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer1_237222*
_output_shapes
:*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer2_237225*
_output_shapes

:*
dtype0?
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer2_237227*
_output_shapes
:*
dtype0?
layer2/bias/Regularizer/SquareSquare5layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer2/bias/Regularizer/SumSum"layer2/bias/Regularizer/Square:y:0&layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/bias/Regularizer/mulMul&layer2/bias/Regularizer/mul/x:output:0$layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_237230*
_output_shapes

:*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpoutput_237232*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/StatefulPartitionedCall.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/StatefulPartitionedCall.^layer2/bias/Regularizer/Square/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2^
-layer2/bias/Regularizer/Square/ReadVariableOp-layer2/bias/Regularizer/Square/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?

?
__inference_loss_fn_1_237830D
6layer0_bias_regularizer_square_readvariableop_resource:
identity??-layer0/bias/Regularizer/Square/ReadVariableOp?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOp6layer0_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
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
?	
?
+__inference_3_layer_NN_layer_call_fn_237037
layer0_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237018o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
??
?
"__inference__traced_restore_238127
file_prefix0
assignvariableop_layer0_kernel:,
assignvariableop_1_layer0_bias:2
 assignvariableop_2_layer1_kernel:,
assignvariableop_3_layer1_bias:2
 assignvariableop_4_layer2_kernel:,
assignvariableop_5_layer2_bias:2
 assignvariableop_6_output_kernel:,
assignvariableop_7_output_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: :
(assignvariableop_17_adam_layer0_kernel_m:4
&assignvariableop_18_adam_layer0_bias_m::
(assignvariableop_19_adam_layer1_kernel_m:4
&assignvariableop_20_adam_layer1_bias_m::
(assignvariableop_21_adam_layer2_kernel_m:4
&assignvariableop_22_adam_layer2_bias_m::
(assignvariableop_23_adam_output_kernel_m:4
&assignvariableop_24_adam_output_bias_m::
(assignvariableop_25_adam_layer0_kernel_v:4
&assignvariableop_26_adam_layer0_bias_v::
(assignvariableop_27_adam_layer1_kernel_v:4
&assignvariableop_28_adam_layer1_bias_v::
(assignvariableop_29_adam_layer2_kernel_v:4
&assignvariableop_30_adam_layer2_bias_v::
(assignvariableop_31_adam_output_kernel_v:4
&assignvariableop_32_adam_output_bias_v:
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
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
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_output_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_output_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_layer0_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_layer0_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_layer1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_layer1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_layer2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_layer2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_output_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_output_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_layer0_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_layer0_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_layer1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_layer1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_layer2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_layer2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_output_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_output_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
?
?
'__inference_layer0_layer_call_fn_237654

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_236877o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_layer2_layer_call_and_return_conditional_losses_237765

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer2/bias/Regularizer/Square/ReadVariableOp?/layer2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer2/bias/Regularizer/SquareSquare5layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer2/bias/Regularizer/SumSum"layer2/bias/Regularizer/Square:y:0&layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/bias/Regularizer/mulMul&layer2/bias/Regularizer/mul/x:output:0$layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer2/bias/Regularizer/Square/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer2/bias/Regularizer/Square/ReadVariableOp-layer2/bias/Regularizer/Square/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?R
?
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237356
layer0_input
layer0_237287:
layer0_237289:
layer1_237292:
layer1_237294:
layer2_237297:
layer2_237299:
output_237302:
output_237304:
identity??layer0/StatefulPartitionedCall?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?layer2/StatefulPartitionedCall?-layer2/bias/Regularizer/Square/ReadVariableOp?/layer2/kernel/Regularizer/Square/ReadVariableOp?output/StatefulPartitionedCall?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/StatefulPartitionedCallStatefulPartitionedCalllayer0_inputlayer0_237287layer0_237289*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_236877?
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_237292layer1_237294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_236906?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_237297layer2_237299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_236935?
output/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0output_237302output_237304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_236963}
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer0_237287*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer0_237289*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_237292*
_output_shapes

:*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer1_237294*
_output_shapes
:*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer2_237297*
_output_shapes

:*
dtype0?
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer2_237299*
_output_shapes
:*
dtype0?
layer2/bias/Regularizer/SquareSquare5layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer2/bias/Regularizer/SumSum"layer2/bias/Regularizer/Square:y:0&layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/bias/Regularizer/mulMul&layer2/bias/Regularizer/mul/x:output:0$layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_237302*
_output_shapes

:*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpoutput_237304*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/StatefulPartitionedCall.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/StatefulPartitionedCall.^layer2/bias/Regularizer/Square/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2^
-layer2/bias/Regularizer/Square/ReadVariableOp-layer2/bias/Regularizer/Square/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?R
?
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237172

inputs
layer0_237103:
layer0_237105:
layer1_237108:
layer1_237110:
layer2_237113:
layer2_237115:
output_237118:
output_237120:
identity??layer0/StatefulPartitionedCall?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?layer2/StatefulPartitionedCall?-layer2/bias/Regularizer/Square/ReadVariableOp?/layer2/kernel/Regularizer/Square/ReadVariableOp?output/StatefulPartitionedCall?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/StatefulPartitionedCallStatefulPartitionedCallinputslayer0_237103layer0_237105*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_236877?
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_237108layer1_237110*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_236906?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_237113layer2_237115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_236935?
output/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0output_237118output_237120*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_236963}
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer0_237103*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer0_237105*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_237108*
_output_shapes

:*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer1_237110*
_output_shapes
:*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer2_237113*
_output_shapes

:*
dtype0?
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer2_237115*
_output_shapes
:*
dtype0?
layer2/bias/Regularizer/SquareSquare5layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer2/bias/Regularizer/SumSum"layer2/bias/Regularizer/Square:y:0&layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/bias/Regularizer/mulMul&layer2/bias/Regularizer/mul/x:output:0$layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_237118*
_output_shapes

:*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpoutput_237120*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/StatefulPartitionedCall.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/StatefulPartitionedCall.^layer2/bias/Regularizer/Square/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2^
-layer2/bias/Regularizer/Square/ReadVariableOp-layer2/bias/Regularizer/Square/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_7_237896D
6output_bias_regularizer_square_readvariableop_resource:
identity??-output/bias/Regularizer/Square/ReadVariableOp?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOp6output_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
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
?
?
B__inference_layer0_layer_call_and_return_conditional_losses_236877

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_3_237852D
6layer1_bias_regularizer_square_readvariableop_resource:
identity??-layer1/bias/Regularizer/Square/ReadVariableOp?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOp6layer1_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
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
+__inference_3_layer_NN_layer_call_fn_237212
layer0_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237172o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?
?
B__inference_output_layer_call_and_return_conditional_losses_237808

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?`
?	
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237610

inputs7
%layer0_matmul_readvariableop_resource:4
&layer0_biasadd_readvariableop_resource:7
%layer1_matmul_readvariableop_resource:4
&layer1_biasadd_readvariableop_resource:7
%layer2_matmul_readvariableop_resource:4
&layer2_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??layer0/BiasAdd/ReadVariableOp?layer0/MatMul/ReadVariableOp?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/MatMul/ReadVariableOp?-layer2/bias/Regularizer/Square/ReadVariableOp?/layer2/kernel/Regularizer/Square/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/MatMul/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0w
layer0/MatMulMatMulinputs$layer0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
layer0/BiasAdd/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/BiasAddBiasAddlayer0/MatMul:product:0%layer0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
layer0/ReluRelulayer0/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
layer1/MatMulMatMullayer0/Relu:activations:0$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output/MatMulMatMullayer2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer2/bias/Regularizer/SquareSquare5layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer2/bias/Regularizer/SumSum"layer2/bias/Regularizer/Square:y:0&layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/bias/Regularizer/mulMul&layer2/bias/Regularizer/mul/x:output:0$layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/BiasAdd/ReadVariableOp^layer0/MatMul/ReadVariableOp.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp.^layer2/bias/Regularizer/Square/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
layer0/BiasAdd/ReadVariableOplayer0/BiasAdd/ReadVariableOp2<
layer0/MatMul/ReadVariableOplayer0/MatMul/ReadVariableOp2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2^
-layer2/bias/Regularizer/Square/ReadVariableOp-layer2/bias/Regularizer/Square/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_layer2_layer_call_fn_237742

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_236935o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_output_layer_call_and_return_conditional_losses_236963

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?`
?	
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237531

inputs7
%layer0_matmul_readvariableop_resource:4
&layer0_biasadd_readvariableop_resource:7
%layer1_matmul_readvariableop_resource:4
&layer1_biasadd_readvariableop_resource:7
%layer2_matmul_readvariableop_resource:4
&layer2_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??layer0/BiasAdd/ReadVariableOp?layer0/MatMul/ReadVariableOp?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/BiasAdd/ReadVariableOp?layer1/MatMul/ReadVariableOp?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?layer2/BiasAdd/ReadVariableOp?layer2/MatMul/ReadVariableOp?-layer2/bias/Regularizer/Square/ReadVariableOp?/layer2/kernel/Regularizer/Square/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/MatMul/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0w
layer0/MatMulMatMulinputs$layer0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
layer0/BiasAdd/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/BiasAddBiasAddlayer0/MatMul:product:0%layer0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
layer0/ReluRelulayer0/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
layer1/MatMulMatMullayer0/Relu:activations:0$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
output/MatMulMatMullayer2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer2/bias/Regularizer/SquareSquare5layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer2/bias/Regularizer/SumSum"layer2/bias/Regularizer/Square:y:0&layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/bias/Regularizer/mulMul&layer2/bias/Regularizer/mul/x:output:0$layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityoutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/BiasAdd/ReadVariableOp^layer0/MatMul/ReadVariableOp.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp.^layer2/bias/Regularizer/Square/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2>
layer0/BiasAdd/ReadVariableOplayer0/BiasAdd/ReadVariableOp2<
layer0/MatMul/ReadVariableOplayer0/MatMul/ReadVariableOp2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2^
-layer2/bias/Regularizer/Square/ReadVariableOp-layer2/bias/Regularizer/Square/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
__inference__traced_save_238018
file_prefix,
(savev2_layer0_kernel_read_readvariableop*
&savev2_layer0_bias_read_readvariableop,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_layer0_kernel_m_read_readvariableop1
-savev2_adam_layer0_bias_m_read_readvariableop3
/savev2_adam_layer1_kernel_m_read_readvariableop1
-savev2_adam_layer1_bias_m_read_readvariableop3
/savev2_adam_layer2_kernel_m_read_readvariableop1
-savev2_adam_layer2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop3
/savev2_adam_layer0_kernel_v_read_readvariableop1
-savev2_adam_layer0_bias_v_read_readvariableop3
/savev2_adam_layer1_kernel_v_read_readvariableop1
-savev2_adam_layer1_bias_v_read_readvariableop3
/savev2_adam_layer2_kernel_v_read_readvariableop1
-savev2_adam_layer2_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer0_kernel_read_readvariableop&savev2_layer0_bias_read_readvariableop(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_layer0_kernel_m_read_readvariableop-savev2_adam_layer0_bias_m_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop/savev2_adam_layer0_kernel_v_read_readvariableop-savev2_adam_layer0_bias_v_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::: : : : : : : : : ::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	
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
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::"

_output_shapes
: 
?	
?
+__inference_3_layer_NN_layer_call_fn_237452

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237172o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_layer2_layer_call_and_return_conditional_losses_236935

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer2/bias/Regularizer/Square/ReadVariableOp?/layer2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer2/bias/Regularizer/SquareSquare5layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer2/bias/Regularizer/SumSum"layer2/bias/Regularizer/Square:y:0&layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/bias/Regularizer/mulMul&layer2/bias/Regularizer/mul/x:output:0$layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer2/bias/Regularizer/Square/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer2/bias/Regularizer/Square/ReadVariableOp-layer2/bias/Regularizer/Square/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_237819J
8layer0_kernel_regularizer_square_readvariableop_resource:
identity??/layer0/kernel/Regularizer/Square/ReadVariableOp?
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8layer0_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
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
?R
?
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237018

inputs
layer0_236878:
layer0_236880:
layer1_236907:
layer1_236909:
layer2_236936:
layer2_236938:
output_236964:
output_236966:
identity??layer0/StatefulPartitionedCall?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOp?layer1/StatefulPartitionedCall?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOp?layer2/StatefulPartitionedCall?-layer2/bias/Regularizer/Square/ReadVariableOp?/layer2/kernel/Regularizer/Square/ReadVariableOp?output/StatefulPartitionedCall?-output/bias/Regularizer/Square/ReadVariableOp?/output/kernel/Regularizer/Square/ReadVariableOp?
layer0/StatefulPartitionedCallStatefulPartitionedCallinputslayer0_236878layer0_236880*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer0_layer_call_and_return_conditional_losses_236877?
layer1/StatefulPartitionedCallStatefulPartitionedCall'layer0/StatefulPartitionedCall:output:0layer1_236907layer1_236909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_236906?
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_236936layer2_236938*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer2_layer_call_and_return_conditional_losses_236935?
output/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0output_236964output_236966*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_236963}
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer0_236878*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer0_236880*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_236907*
_output_shapes

:*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer1_236909*
_output_shapes
:*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer2_236936*
_output_shapes

:*
dtype0?
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOplayer2_236938*
_output_shapes
:*
dtype0?
layer2/bias/Regularizer/SquareSquare5layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer2/bias/Regularizer/SumSum"layer2/bias/Regularizer/Square:y:0&layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/bias/Regularizer/mulMul&layer2/bias/Regularizer/mul/x:output:0$layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: }
/output/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_236964*
_output_shapes

:*
dtype0?
 output/kernel/Regularizer/SquareSquare7output/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
output/kernel/Regularizer/mulMul(output/kernel/Regularizer/mul/x:output:0&output/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
-output/bias/Regularizer/Square/ReadVariableOpReadVariableOpoutput_236966*
_output_shapes
:*
dtype0?
output/bias/Regularizer/SquareSquare5output/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
output/bias/Regularizer/mulMul&output/bias/Regularizer/mul/x:output:0$output/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^layer0/StatefulPartitionedCall.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp^layer1/StatefulPartitionedCall.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/StatefulPartitionedCall.^layer2/bias/Regularizer/Square/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp^output/StatefulPartitionedCall.^output/bias/Regularizer/Square/ReadVariableOp0^output/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2@
layer0/StatefulPartitionedCalllayer0/StatefulPartitionedCall2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2^
-layer2/bias/Regularizer/Square/ReadVariableOp-layer2/bias/Regularizer/Square/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2^
-output/bias/Regularizer/Square/ReadVariableOp-output/bias/Regularizer/Square/ReadVariableOp2b
/output/kernel/Regularizer/Square/ReadVariableOp/output/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_layer1_layer_call_fn_237698

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_layer1_layer_call_and_return_conditional_losses_236906o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_output_layer_call_fn_237786

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_236963o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
+__inference_3_layer_NN_layer_call_fn_237431

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237018o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
__inference_loss_fn_5_237874D
6layer2_bias_regularizer_square_readvariableop_resource:
identity??-layer2/bias/Regularizer/Square/ReadVariableOp?
-layer2/bias/Regularizer/Square/ReadVariableOpReadVariableOp6layer2_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0?
layer2/bias/Regularizer/SquareSquare5layer2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
layer2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
layer2/bias/Regularizer/SumSum"layer2/bias/Regularizer/Square:y:0&layer2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: b
layer2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??8?
layer2/bias/Regularizer/mulMul&layer2/bias/Regularizer/mul/x:output:0$layer2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ]
IdentityIdentitylayer2/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: v
NoOpNoOp.^layer2/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-layer2/bias/Regularizer/Square/ReadVariableOp-layer2/bias/Regularizer/Square/ReadVariableOp
?)
?
!__inference__wrapped_model_236847
layer0_input@
.layer_nn_layer0_matmul_readvariableop_resource:=
/layer_nn_layer0_biasadd_readvariableop_resource:@
.layer_nn_layer1_matmul_readvariableop_resource:=
/layer_nn_layer1_biasadd_readvariableop_resource:@
.layer_nn_layer2_matmul_readvariableop_resource:=
/layer_nn_layer2_biasadd_readvariableop_resource:@
.layer_nn_output_matmul_readvariableop_resource:=
/layer_nn_output_biasadd_readvariableop_resource:
identity??(3_layer_NN/layer0/BiasAdd/ReadVariableOp?'3_layer_NN/layer0/MatMul/ReadVariableOp?(3_layer_NN/layer1/BiasAdd/ReadVariableOp?'3_layer_NN/layer1/MatMul/ReadVariableOp?(3_layer_NN/layer2/BiasAdd/ReadVariableOp?'3_layer_NN/layer2/MatMul/ReadVariableOp?(3_layer_NN/output/BiasAdd/ReadVariableOp?'3_layer_NN/output/MatMul/ReadVariableOp?
'3_layer_NN/layer0/MatMul/ReadVariableOpReadVariableOp.layer_nn_layer0_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
3_layer_NN/layer0/MatMulMatMullayer0_input/3_layer_NN/layer0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(3_layer_NN/layer0/BiasAdd/ReadVariableOpReadVariableOp/layer_nn_layer0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
3_layer_NN/layer0/BiasAddBiasAdd"3_layer_NN/layer0/MatMul:product:003_layer_NN/layer0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
3_layer_NN/layer0/ReluRelu"3_layer_NN/layer0/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'3_layer_NN/layer1/MatMul/ReadVariableOpReadVariableOp.layer_nn_layer1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
3_layer_NN/layer1/MatMulMatMul$3_layer_NN/layer0/Relu:activations:0/3_layer_NN/layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(3_layer_NN/layer1/BiasAdd/ReadVariableOpReadVariableOp/layer_nn_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
3_layer_NN/layer1/BiasAddBiasAdd"3_layer_NN/layer1/MatMul:product:003_layer_NN/layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
3_layer_NN/layer1/ReluRelu"3_layer_NN/layer1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'3_layer_NN/layer2/MatMul/ReadVariableOpReadVariableOp.layer_nn_layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
3_layer_NN/layer2/MatMulMatMul$3_layer_NN/layer1/Relu:activations:0/3_layer_NN/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(3_layer_NN/layer2/BiasAdd/ReadVariableOpReadVariableOp/layer_nn_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
3_layer_NN/layer2/BiasAddBiasAdd"3_layer_NN/layer2/MatMul:product:003_layer_NN/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
3_layer_NN/layer2/ReluRelu"3_layer_NN/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'3_layer_NN/output/MatMul/ReadVariableOpReadVariableOp.layer_nn_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
3_layer_NN/output/MatMulMatMul$3_layer_NN/layer2/Relu:activations:0/3_layer_NN/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(3_layer_NN/output/BiasAdd/ReadVariableOpReadVariableOp/layer_nn_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
3_layer_NN/output/BiasAddBiasAdd"3_layer_NN/output/MatMul:product:003_layer_NN/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"3_layer_NN/output/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^3_layer_NN/layer0/BiasAdd/ReadVariableOp(^3_layer_NN/layer0/MatMul/ReadVariableOp)^3_layer_NN/layer1/BiasAdd/ReadVariableOp(^3_layer_NN/layer1/MatMul/ReadVariableOp)^3_layer_NN/layer2/BiasAdd/ReadVariableOp(^3_layer_NN/layer2/MatMul/ReadVariableOp)^3_layer_NN/output/BiasAdd/ReadVariableOp(^3_layer_NN/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 2T
(3_layer_NN/layer0/BiasAdd/ReadVariableOp(3_layer_NN/layer0/BiasAdd/ReadVariableOp2R
'3_layer_NN/layer0/MatMul/ReadVariableOp'3_layer_NN/layer0/MatMul/ReadVariableOp2T
(3_layer_NN/layer1/BiasAdd/ReadVariableOp(3_layer_NN/layer1/BiasAdd/ReadVariableOp2R
'3_layer_NN/layer1/MatMul/ReadVariableOp'3_layer_NN/layer1/MatMul/ReadVariableOp2T
(3_layer_NN/layer2/BiasAdd/ReadVariableOp(3_layer_NN/layer2/BiasAdd/ReadVariableOp2R
'3_layer_NN/layer2/MatMul/ReadVariableOp'3_layer_NN/layer2/MatMul/ReadVariableOp2T
(3_layer_NN/output/BiasAdd/ReadVariableOp(3_layer_NN/output/BiasAdd/ReadVariableOp2R
'3_layer_NN/output/MatMul/ReadVariableOp'3_layer_NN/output/MatMul/ReadVariableOp:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?
?
B__inference_layer1_layer_call_and_return_conditional_losses_237721

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer1/bias/Regularizer/Square/ReadVariableOp?/layer1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer1/bias/Regularizer/SquareSquare5layer1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer1/bias/Regularizer/mulMul&layer1/bias/Regularizer/mul/x:output:0$layer1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer1/bias/Regularizer/Square/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer1/bias/Regularizer/Square/ReadVariableOp-layer1/bias/Regularizer/Square/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_237633
layer0_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_236847o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_namelayer0_input
?
?
B__inference_layer0_layer_call_and_return_conditional_losses_237677

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-layer0/bias/Regularizer/Square/ReadVariableOp?/layer0/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
/layer0/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0?
 layer0/kernel/Regularizer/SquareSquare7layer0/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
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
 *??8?
layer0/kernel/Regularizer/mulMul(layer0/kernel/Regularizer/mul/x:output:0&layer0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
-layer0/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
layer0/bias/Regularizer/SquareSquare5layer0/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:g
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
 *??8?
layer0/bias/Regularizer/mulMul&layer0/bias/Regularizer/mul/x:output:0$layer0/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^layer0/bias/Regularizer/Square/ReadVariableOp0^layer0/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-layer0/bias/Regularizer/Square/ReadVariableOp-layer0/bias/Regularizer/Square/ReadVariableOp2b
/layer0/kernel/Regularizer/Square/ReadVariableOp/layer0/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
serving_default_layer0_input:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?m
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
?
.iter

/beta_1

0beta_2
	1decay
2learning_ratem`mambmcmdme&mf'mgvhvivjvkvlvm&vn'vo"
	optimizer
X
0
1
2
3
4
5
&6
'7"
trackable_list_wrapper
X
0
1
2
3
4
5
&6
'7"
trackable_list_wrapper
X
30
41
52
63
74
85
96
:7"
trackable_list_wrapper
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_3_layer_NN_layer_call_fn_237037
+__inference_3_layer_NN_layer_call_fn_237431
+__inference_3_layer_NN_layer_call_fn_237452
+__inference_3_layer_NN_layer_call_fn_237212?
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
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237531
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237610
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237284
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237356?
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
!__inference__wrapped_model_236847layer0_input"?
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
@serving_default"
signature_map
:2layer0/kernel
:2layer0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_layer0_layer_call_fn_237654?
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
B__inference_layer0_layer_call_and_return_conditional_losses_237677?
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
:2layer1/kernel
:2layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_layer1_layer_call_fn_237698?
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
B__inference_layer1_layer_call_and_return_conditional_losses_237721?
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
:2layer2/kernel
:2layer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_layer2_layer_call_fn_237742?
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
B__inference_layer2_layer_call_and_return_conditional_losses_237765?
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
:2output/kernel
:2output/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_output_layer_call_fn_237786?
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
B__inference_output_layer_call_and_return_conditional_losses_237808?
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?2?
__inference_loss_fn_0_237819?
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
__inference_loss_fn_1_237830?
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
__inference_loss_fn_2_237841?
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
__inference_loss_fn_3_237852?
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
__inference_loss_fn_4_237863?
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
__inference_loss_fn_5_237874?
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
__inference_loss_fn_6_237885?
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
__inference_loss_fn_7_237896?
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
<
0
1
2
3"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_237633layer0_input"?
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
30
41"
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
50
61"
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
70
81"
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
90
:1"
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Wtotal
	Xcount
Y	variables
Z	keras_api"
_tf_keras_metric
^
	[total
	\count
]
_fn_kwargs
^	variables
_	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
W0
X1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
[0
\1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
$:"2Adam/layer0/kernel/m
:2Adam/layer0/bias/m
$:"2Adam/layer1/kernel/m
:2Adam/layer1/bias/m
$:"2Adam/layer2/kernel/m
:2Adam/layer2/bias/m
$:"2Adam/output/kernel/m
:2Adam/output/bias/m
$:"2Adam/layer0/kernel/v
:2Adam/layer0/bias/v
$:"2Adam/layer1/kernel/v
:2Adam/layer1/bias/v
$:"2Adam/layer2/kernel/v
:2Adam/layer2/bias/v
$:"2Adam/output/kernel/v
:2Adam/output/bias/v?
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237284p&'=?:
3?0
&?#
layer0_input?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237356p&'=?:
3?0
&?#
layer0_input?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237531j&'7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_3_layer_NN_layer_call_and_return_conditional_losses_237610j&'7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
+__inference_3_layer_NN_layer_call_fn_237037c&'=?:
3?0
&?#
layer0_input?????????
p 

 
? "???????????
+__inference_3_layer_NN_layer_call_fn_237212c&'=?:
3?0
&?#
layer0_input?????????
p

 
? "???????????
+__inference_3_layer_NN_layer_call_fn_237431]&'7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
+__inference_3_layer_NN_layer_call_fn_237452]&'7?4
-?*
 ?
inputs?????????
p

 
? "???????????
!__inference__wrapped_model_236847r&'5?2
+?(
&?#
layer0_input?????????
? "/?,
*
output ?
output??????????
B__inference_layer0_layer_call_and_return_conditional_losses_237677\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_layer0_layer_call_fn_237654O/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_layer1_layer_call_and_return_conditional_losses_237721\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_layer1_layer_call_fn_237698O/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_layer2_layer_call_and_return_conditional_losses_237765\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_layer2_layer_call_fn_237742O/?,
%?"
 ?
inputs?????????
? "??????????;
__inference_loss_fn_0_237819?

? 
? "? ;
__inference_loss_fn_1_237830?

? 
? "? ;
__inference_loss_fn_2_237841?

? 
? "? ;
__inference_loss_fn_3_237852?

? 
? "? ;
__inference_loss_fn_4_237863?

? 
? "? ;
__inference_loss_fn_5_237874?

? 
? "? ;
__inference_loss_fn_6_237885&?

? 
? "? ;
__inference_loss_fn_7_237896'?

? 
? "? ?
B__inference_output_layer_call_and_return_conditional_losses_237808\&'/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_output_layer_call_fn_237786O&'/?,
%?"
 ?
inputs?????????
? "???????????
$__inference_signature_wrapper_237633?&'E?B
? 
;?8
6
layer0_input&?#
layer0_input?????????"/?,
*
output ?
output?????????