ýÿ
°
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Û·

ctrl_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*$
shared_namectrl_layer_1/kernel
|
'ctrl_layer_1/kernel/Read/ReadVariableOpReadVariableOpctrl_layer_1/kernel*
_output_shapes
:	(*
dtype0
{
ctrl_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namectrl_layer_1/bias
t
%ctrl_layer_1/bias/Read/ReadVariableOpReadVariableOpctrl_layer_1/bias*
_output_shapes	
:*
dtype0

ctrl_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namectrl_layer_2/kernel
}
'ctrl_layer_2/kernel/Read/ReadVariableOpReadVariableOpctrl_layer_2/kernel* 
_output_shapes
:
*
dtype0
{
ctrl_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namectrl_layer_2/bias
t
%ctrl_layer_2/bias/Read/ReadVariableOpReadVariableOpctrl_layer_2/bias*
_output_shapes	
:*
dtype0

ctrl_layer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_namectrl_layer_3/kernel
|
'ctrl_layer_3/kernel/Read/ReadVariableOpReadVariableOpctrl_layer_3/kernel*
_output_shapes
:	*
dtype0
z
ctrl_layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namectrl_layer_3/bias
s
%ctrl_layer_3/bias/Read/ReadVariableOpReadVariableOpctrl_layer_3/bias*
_output_shapes
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0

pred_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	)*$
shared_namepred_layer_1/kernel
|
'pred_layer_1/kernel/Read/ReadVariableOpReadVariableOppred_layer_1/kernel*
_output_shapes
:	)*
dtype0
{
pred_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namepred_layer_1/bias
t
%pred_layer_1/bias/Read/ReadVariableOpReadVariableOppred_layer_1/bias*
_output_shapes	
:*
dtype0

pred_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_namepred_layer_2/kernel
}
'pred_layer_2/kernel/Read/ReadVariableOpReadVariableOppred_layer_2/kernel* 
_output_shapes
:
*
dtype0
{
pred_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namepred_layer_2/bias
t
%pred_layer_2/bias/Read/ReadVariableOpReadVariableOppred_layer_2/bias*
_output_shapes	
:*
dtype0

pred_layer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_namepred_layer_3/kernel
|
'pred_layer_3/kernel/Read/ReadVariableOpReadVariableOppred_layer_3/kernel*
_output_shapes
:	*
dtype0
z
pred_layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namepred_layer_3/bias
s
%pred_layer_3/bias/Read/ReadVariableOpReadVariableOppred_layer_3/bias*
_output_shapes
:*
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
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0

NoOpNoOp
ÀE
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ûD
valueñDBîD BçD
ò
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

loss
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
¦

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses*
Õ
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*

6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
¦

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses*
¦

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
¦

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses*
* 
z
0
1
2
3
#4
$5
,6
-7
.8
/9
<10
=11
D12
E13
L14
M15*
j
0
1
2
3
#4
$5
,6
-7
<8
=9
D10
E11
L12
M13*
X
T0
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11* 
°
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

eserving_default* 
c]
VARIABLE_VALUEctrl_layer_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEctrl_layer_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

T0
U1* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEctrl_layer_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEctrl_layer_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*

V0
W1* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEctrl_layer_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEctrl_layer_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*

X0
Y1* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
* 
* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
,0
-1
.2
/3*

,0
-1*
* 

unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 
* 
* 
c]
VARIABLE_VALUEpred_layer_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEpred_layer_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

<0
=1*

<0
=1*

Z0
[1* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEpred_layer_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEpred_layer_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

D0
E1*

\0
]1* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUEpred_layer_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEpred_layer_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

L0
M1*

^0
_1* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

.0
/1*
C
0
1
2
3
4
5
6
7
	8*
,
0
1
2
3
4*
* 
* 
* 
* 
* 
* 

T0
U1* 
* 
* 
* 
* 

V0
W1* 
* 
* 
* 
* 

X0
Y1* 
* 

.0
/1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Z0
[1* 
* 
* 
* 
* 

\0
]1* 
* 
* 
* 
* 

^0
_1* 
* 
<

total

count
	variables
	keras_api*
<

total

count
	variables
	keras_api*
<

total

count
	variables
	keras_api*
M

total

 count
¡
_fn_kwargs
¢	variables
£	keras_api*
M

¤total

¥count
¦
_fn_kwargs
§	variables
¨	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
 1*

¢	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¤0
¥1*

§	variables*
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ(
Ñ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1ctrl_layer_1/kernelctrl_layer_1/biasctrl_layer_2/kernelctrl_layer_2/biasctrl_layer_3/kernelctrl_layer_3/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betapred_layer_1/kernelpred_layer_1/biaspred_layer_2/kernelpred_layer_2/biaspred_layer_3/kernelpred_layer_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_121557
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
þ	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'ctrl_layer_1/kernel/Read/ReadVariableOp%ctrl_layer_1/bias/Read/ReadVariableOp'ctrl_layer_2/kernel/Read/ReadVariableOp%ctrl_layer_2/bias/Read/ReadVariableOp'ctrl_layer_3/kernel/Read/ReadVariableOp%ctrl_layer_3/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp'pred_layer_1/kernel/Read/ReadVariableOp%pred_layer_1/bias/Read/ReadVariableOp'pred_layer_2/kernel/Read/ReadVariableOp%pred_layer_2/bias/Read/ReadVariableOp'pred_layer_3/kernel/Read/ReadVariableOp%pred_layer_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOpConst*'
Tin 
2*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_122148
ñ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamectrl_layer_1/kernelctrl_layer_1/biasctrl_layer_2/kernelctrl_layer_2/biasctrl_layer_3/kernelctrl_layer_3/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancepred_layer_1/kernelpred_layer_1/biaspred_layer_2/kernelpred_layer_2/biaspred_layer_3/kernelpred_layer_3/biastotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4*&
Tin
2*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_122236«
ï

A__inference_model_layer_call_and_return_conditional_losses_120506

inputs&
ctrl_layer_1_120264:	("
ctrl_layer_1_120266:	'
ctrl_layer_2_120293:
"
ctrl_layer_2_120295:	&
ctrl_layer_3_120322:	!
ctrl_layer_3_120324:(
batch_normalization_120327:(
batch_normalization_120329:(
batch_normalization_120331:(
batch_normalization_120333:&
pred_layer_1_120369:	)"
pred_layer_1_120371:	'
pred_layer_2_120398:
"
pred_layer_2_120400:	&
pred_layer_3_120427:	!
pred_layer_3_120429:
identity

identity_1¢+batch_normalization/StatefulPartitionedCall¢$ctrl_layer_1/StatefulPartitionedCall¢3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp¢$ctrl_layer_2/StatefulPartitionedCall¢3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp¢$ctrl_layer_3/StatefulPartitionedCall¢3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_1/StatefulPartitionedCall¢3pred_layer_1/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_2/StatefulPartitionedCall¢3pred_layer_2/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_3/StatefulPartitionedCall¢3pred_layer_3/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp
$ctrl_layer_1/StatefulPartitionedCallStatefulPartitionedCallinputsctrl_layer_1_120264ctrl_layer_1_120266*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_1_layer_call_and_return_conditional_losses_120263¨
$ctrl_layer_2/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_1/StatefulPartitionedCall:output:0ctrl_layer_2_120293ctrl_layer_2_120295*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_2_layer_call_and_return_conditional_losses_120292§
$ctrl_layer_3/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_2/StatefulPartitionedCall:output:0ctrl_layer_3_120322ctrl_layer_3_120324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_3_layer_call_and_return_conditional_losses_120321ÿ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_3/StatefulPartitionedCall:output:0batch_normalization_120327batch_normalization_120329batch_normalization_120331batch_normalization_120333*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_120175õ
concatenate/PartitionedCallPartitionedCallinputs4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_120343
$pred_layer_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0pred_layer_1_120369pred_layer_1_120371*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_1_layer_call_and_return_conditional_losses_120368¨
$pred_layer_2/StatefulPartitionedCallStatefulPartitionedCall-pred_layer_1/StatefulPartitionedCall:output:0pred_layer_2_120398pred_layer_2_120400*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_2_layer_call_and_return_conditional_losses_120397§
$pred_layer_3/StatefulPartitionedCallStatefulPartitionedCall-pred_layer_2/StatefulPartitionedCall:output:0pred_layer_3_120427pred_layer_3_120429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_3_layer_call_and_return_conditional_losses_120426
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_1_120264*
_output_shapes
:	(*
dtype0
&ctrl_layer_1/kernel/Regularizer/SquareSquare=ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	(v
%ctrl_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_1/kernel/Regularizer/SumSum*ctrl_layer_1/kernel/Regularizer/Square:y:0.ctrl_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_1/kernel/Regularizer/mulMul.ctrl_layer_1/kernel/Regularizer/mul/x:output:0,ctrl_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_1_120266*
_output_shapes	
:*
dtype0
$ctrl_layer_1/bias/Regularizer/SquareSquare;ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_1/bias/Regularizer/SumSum(ctrl_layer_1/bias/Regularizer/Square:y:0,ctrl_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_1/bias/Regularizer/mulMul,ctrl_layer_1/bias/Regularizer/mul/x:output:0*ctrl_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_2_120293* 
_output_shapes
:
*
dtype0
&ctrl_layer_2/kernel/Regularizer/SquareSquare=ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%ctrl_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_2/kernel/Regularizer/SumSum*ctrl_layer_2/kernel/Regularizer/Square:y:0.ctrl_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_2/kernel/Regularizer/mulMul.ctrl_layer_2/kernel/Regularizer/mul/x:output:0,ctrl_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_2_120295*
_output_shapes	
:*
dtype0
$ctrl_layer_2/bias/Regularizer/SquareSquare;ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_2/bias/Regularizer/SumSum(ctrl_layer_2/bias/Regularizer/Square:y:0,ctrl_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_2/bias/Regularizer/mulMul,ctrl_layer_2/bias/Regularizer/mul/x:output:0*ctrl_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_3_120322*
_output_shapes
:	*
dtype0
&ctrl_layer_3/kernel/Regularizer/SquareSquare=ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%ctrl_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_3/kernel/Regularizer/SumSum*ctrl_layer_3/kernel/Regularizer/Square:y:0.ctrl_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_3/kernel/Regularizer/mulMul.ctrl_layer_3/kernel/Regularizer/mul/x:output:0,ctrl_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_3_120324*
_output_shapes
:*
dtype0
$ctrl_layer_3/bias/Regularizer/SquareSquare;ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#ctrl_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_3/bias/Regularizer/SumSum(ctrl_layer_3/bias/Regularizer/Square:y:0,ctrl_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_3/bias/Regularizer/mulMul,ctrl_layer_3/bias/Regularizer/mul/x:output:0*ctrl_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_1_120369*
_output_shapes
:	)*
dtype0
&pred_layer_1/kernel/Regularizer/SquareSquare=pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	)v
%pred_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_1/kernel/Regularizer/SumSum*pred_layer_1/kernel/Regularizer/Square:y:0.pred_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_1/kernel/Regularizer/mulMul.pred_layer_1/kernel/Regularizer/mul/x:output:0,pred_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_1_120371*
_output_shapes	
:*
dtype0
$pred_layer_1/bias/Regularizer/SquareSquare;pred_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_1/bias/Regularizer/SumSum(pred_layer_1/bias/Regularizer/Square:y:0,pred_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_1/bias/Regularizer/mulMul,pred_layer_1/bias/Regularizer/mul/x:output:0*pred_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_2_120398* 
_output_shapes
:
*
dtype0
&pred_layer_2/kernel/Regularizer/SquareSquare=pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%pred_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_2/kernel/Regularizer/SumSum*pred_layer_2/kernel/Regularizer/Square:y:0.pred_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_2/kernel/Regularizer/mulMul.pred_layer_2/kernel/Regularizer/mul/x:output:0,pred_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_2_120400*
_output_shapes	
:*
dtype0
$pred_layer_2/bias/Regularizer/SquareSquare;pred_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_2/bias/Regularizer/SumSum(pred_layer_2/bias/Regularizer/Square:y:0,pred_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_2/bias/Regularizer/mulMul,pred_layer_2/bias/Regularizer/mul/x:output:0*pred_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_3_120427*
_output_shapes
:	*
dtype0
&pred_layer_3/kernel/Regularizer/SquareSquare=pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%pred_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_3/kernel/Regularizer/SumSum*pred_layer_3/kernel/Regularizer/Square:y:0.pred_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_3/kernel/Regularizer/mulMul.pred_layer_3/kernel/Regularizer/mul/x:output:0,pred_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_3_120429*
_output_shapes
:*
dtype0
$pred_layer_3/bias/Regularizer/SquareSquare;pred_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#pred_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_3/bias/Regularizer/SumSum(pred_layer_3/bias/Regularizer/Square:y:0,pred_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_3/bias/Regularizer/mulMul,pred_layer_3/bias/Regularizer/mul/x:output:0*pred_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-pred_layer_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~

Identity_1Identity-ctrl_layer_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOp,^batch_normalization/StatefulPartitionedCall%^ctrl_layer_1/StatefulPartitionedCall4^ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp%^ctrl_layer_2/StatefulPartitionedCall4^ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp%^ctrl_layer_3/StatefulPartitionedCall4^ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_1/StatefulPartitionedCall4^pred_layer_1/bias/Regularizer/Square/ReadVariableOp6^pred_layer_1/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_2/StatefulPartitionedCall4^pred_layer_2/bias/Regularizer/Square/ReadVariableOp6^pred_layer_2/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_3/StatefulPartitionedCall4^pred_layer_3/bias/Regularizer/Square/ReadVariableOp6^pred_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$ctrl_layer_1/StatefulPartitionedCall$ctrl_layer_1/StatefulPartitionedCall2j
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp2L
$ctrl_layer_2/StatefulPartitionedCall$ctrl_layer_2/StatefulPartitionedCall2j
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$ctrl_layer_3/StatefulPartitionedCall$ctrl_layer_3/StatefulPartitionedCall2j
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_1/StatefulPartitionedCall$pred_layer_1/StatefulPartitionedCall2j
3pred_layer_1/bias/Regularizer/Square/ReadVariableOp3pred_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_2/StatefulPartitionedCall$pred_layer_2/StatefulPartitionedCall2j
3pred_layer_2/bias/Regularizer/Square/ReadVariableOp3pred_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_3/StatefulPartitionedCall$pred_layer_3/StatefulPartitionedCall2j
3pred_layer_3/bias/Regularizer/Square/ReadVariableOp3pred_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
È
±
__inference_loss_fn_5_121980J
<ctrl_layer_3_bias_regularizer_square_readvariableop_resource:
identity¢3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp¬
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp<ctrl_layer_3_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0
$ctrl_layer_3/bias/Regularizer/SquareSquare;ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#ctrl_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_3/bias/Regularizer/SumSum(ctrl_layer_3/bias/Regularizer/Square:y:0,ctrl_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_3/bias/Regularizer/mulMul,ctrl_layer_3/bias/Regularizer/mul/x:output:0*ctrl_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%ctrl_layer_3/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp
Î
¦
&__inference_model_layer_call_fn_121228

inputs
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	)

unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_120768o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

º
__inference_loss_fn_4_121969Q
>ctrl_layer_3_kernel_regularizer_square_readvariableop_resource:	
identity¢5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpµ
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>ctrl_layer_3_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype0
&ctrl_layer_3/kernel/Regularizer/SquareSquare=ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%ctrl_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_3/kernel/Regularizer/SumSum*ctrl_layer_3/kernel/Regularizer/Square:y:0.ctrl_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_3/kernel/Regularizer/mulMul.ctrl_layer_3/kernel/Regularizer/mul/x:output:0,ctrl_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentity'ctrl_layer_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp
À
s
G__inference_concatenate_layer_call_and_return_conditional_losses_121782
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ó

A__inference_model_layer_call_and_return_conditional_losses_120961
input_1&
ctrl_layer_1_120847:	("
ctrl_layer_1_120849:	'
ctrl_layer_2_120852:
"
ctrl_layer_2_120854:	&
ctrl_layer_3_120857:	!
ctrl_layer_3_120859:(
batch_normalization_120862:(
batch_normalization_120864:(
batch_normalization_120866:(
batch_normalization_120868:&
pred_layer_1_120872:	)"
pred_layer_1_120874:	'
pred_layer_2_120877:
"
pred_layer_2_120879:	&
pred_layer_3_120882:	!
pred_layer_3_120884:
identity

identity_1¢+batch_normalization/StatefulPartitionedCall¢$ctrl_layer_1/StatefulPartitionedCall¢3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp¢$ctrl_layer_2/StatefulPartitionedCall¢3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp¢$ctrl_layer_3/StatefulPartitionedCall¢3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_1/StatefulPartitionedCall¢3pred_layer_1/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_2/StatefulPartitionedCall¢3pred_layer_2/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_3/StatefulPartitionedCall¢3pred_layer_3/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp
$ctrl_layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_1ctrl_layer_1_120847ctrl_layer_1_120849*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_1_layer_call_and_return_conditional_losses_120263¨
$ctrl_layer_2/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_1/StatefulPartitionedCall:output:0ctrl_layer_2_120852ctrl_layer_2_120854*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_2_layer_call_and_return_conditional_losses_120292§
$ctrl_layer_3/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_2/StatefulPartitionedCall:output:0ctrl_layer_3_120857ctrl_layer_3_120859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_3_layer_call_and_return_conditional_losses_120321ÿ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_3/StatefulPartitionedCall:output:0batch_normalization_120862batch_normalization_120864batch_normalization_120866batch_normalization_120868*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_120175ö
concatenate/PartitionedCallPartitionedCallinput_14batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_120343
$pred_layer_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0pred_layer_1_120872pred_layer_1_120874*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_1_layer_call_and_return_conditional_losses_120368¨
$pred_layer_2/StatefulPartitionedCallStatefulPartitionedCall-pred_layer_1/StatefulPartitionedCall:output:0pred_layer_2_120877pred_layer_2_120879*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_2_layer_call_and_return_conditional_losses_120397§
$pred_layer_3/StatefulPartitionedCallStatefulPartitionedCall-pred_layer_2/StatefulPartitionedCall:output:0pred_layer_3_120882pred_layer_3_120884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_3_layer_call_and_return_conditional_losses_120426
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_1_120847*
_output_shapes
:	(*
dtype0
&ctrl_layer_1/kernel/Regularizer/SquareSquare=ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	(v
%ctrl_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_1/kernel/Regularizer/SumSum*ctrl_layer_1/kernel/Regularizer/Square:y:0.ctrl_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_1/kernel/Regularizer/mulMul.ctrl_layer_1/kernel/Regularizer/mul/x:output:0,ctrl_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_1_120849*
_output_shapes	
:*
dtype0
$ctrl_layer_1/bias/Regularizer/SquareSquare;ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_1/bias/Regularizer/SumSum(ctrl_layer_1/bias/Regularizer/Square:y:0,ctrl_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_1/bias/Regularizer/mulMul,ctrl_layer_1/bias/Regularizer/mul/x:output:0*ctrl_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_2_120852* 
_output_shapes
:
*
dtype0
&ctrl_layer_2/kernel/Regularizer/SquareSquare=ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%ctrl_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_2/kernel/Regularizer/SumSum*ctrl_layer_2/kernel/Regularizer/Square:y:0.ctrl_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_2/kernel/Regularizer/mulMul.ctrl_layer_2/kernel/Regularizer/mul/x:output:0,ctrl_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_2_120854*
_output_shapes	
:*
dtype0
$ctrl_layer_2/bias/Regularizer/SquareSquare;ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_2/bias/Regularizer/SumSum(ctrl_layer_2/bias/Regularizer/Square:y:0,ctrl_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_2/bias/Regularizer/mulMul,ctrl_layer_2/bias/Regularizer/mul/x:output:0*ctrl_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_3_120857*
_output_shapes
:	*
dtype0
&ctrl_layer_3/kernel/Regularizer/SquareSquare=ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%ctrl_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_3/kernel/Regularizer/SumSum*ctrl_layer_3/kernel/Regularizer/Square:y:0.ctrl_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_3/kernel/Regularizer/mulMul.ctrl_layer_3/kernel/Regularizer/mul/x:output:0,ctrl_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_3_120859*
_output_shapes
:*
dtype0
$ctrl_layer_3/bias/Regularizer/SquareSquare;ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#ctrl_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_3/bias/Regularizer/SumSum(ctrl_layer_3/bias/Regularizer/Square:y:0,ctrl_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_3/bias/Regularizer/mulMul,ctrl_layer_3/bias/Regularizer/mul/x:output:0*ctrl_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_1_120872*
_output_shapes
:	)*
dtype0
&pred_layer_1/kernel/Regularizer/SquareSquare=pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	)v
%pred_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_1/kernel/Regularizer/SumSum*pred_layer_1/kernel/Regularizer/Square:y:0.pred_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_1/kernel/Regularizer/mulMul.pred_layer_1/kernel/Regularizer/mul/x:output:0,pred_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_1_120874*
_output_shapes	
:*
dtype0
$pred_layer_1/bias/Regularizer/SquareSquare;pred_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_1/bias/Regularizer/SumSum(pred_layer_1/bias/Regularizer/Square:y:0,pred_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_1/bias/Regularizer/mulMul,pred_layer_1/bias/Regularizer/mul/x:output:0*pred_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_2_120877* 
_output_shapes
:
*
dtype0
&pred_layer_2/kernel/Regularizer/SquareSquare=pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%pred_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_2/kernel/Regularizer/SumSum*pred_layer_2/kernel/Regularizer/Square:y:0.pred_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_2/kernel/Regularizer/mulMul.pred_layer_2/kernel/Regularizer/mul/x:output:0,pred_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_2_120879*
_output_shapes	
:*
dtype0
$pred_layer_2/bias/Regularizer/SquareSquare;pred_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_2/bias/Regularizer/SumSum(pred_layer_2/bias/Regularizer/Square:y:0,pred_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_2/bias/Regularizer/mulMul,pred_layer_2/bias/Regularizer/mul/x:output:0*pred_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_3_120882*
_output_shapes
:	*
dtype0
&pred_layer_3/kernel/Regularizer/SquareSquare=pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%pred_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_3/kernel/Regularizer/SumSum*pred_layer_3/kernel/Regularizer/Square:y:0.pred_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_3/kernel/Regularizer/mulMul.pred_layer_3/kernel/Regularizer/mul/x:output:0,pred_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_3_120884*
_output_shapes
:*
dtype0
$pred_layer_3/bias/Regularizer/SquareSquare;pred_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#pred_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_3/bias/Regularizer/SumSum(pred_layer_3/bias/Regularizer/Square:y:0,pred_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_3/bias/Regularizer/mulMul,pred_layer_3/bias/Regularizer/mul/x:output:0*pred_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-pred_layer_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~

Identity_1Identity-ctrl_layer_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOp,^batch_normalization/StatefulPartitionedCall%^ctrl_layer_1/StatefulPartitionedCall4^ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp%^ctrl_layer_2/StatefulPartitionedCall4^ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp%^ctrl_layer_3/StatefulPartitionedCall4^ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_1/StatefulPartitionedCall4^pred_layer_1/bias/Regularizer/Square/ReadVariableOp6^pred_layer_1/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_2/StatefulPartitionedCall4^pred_layer_2/bias/Regularizer/Square/ReadVariableOp6^pred_layer_2/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_3/StatefulPartitionedCall4^pred_layer_3/bias/Regularizer/Square/ReadVariableOp6^pred_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$ctrl_layer_1/StatefulPartitionedCall$ctrl_layer_1/StatefulPartitionedCall2j
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp2L
$ctrl_layer_2/StatefulPartitionedCall$ctrl_layer_2/StatefulPartitionedCall2j
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$ctrl_layer_3/StatefulPartitionedCall$ctrl_layer_3/StatefulPartitionedCall2j
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_1/StatefulPartitionedCall$pred_layer_1/StatefulPartitionedCall2j
3pred_layer_1/bias/Regularizer/Square/ReadVariableOp3pred_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_2/StatefulPartitionedCall$pred_layer_2/StatefulPartitionedCall2j
3pred_layer_2/bias/Regularizer/Square/ReadVariableOp3pred_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_3/StatefulPartitionedCall$pred_layer_3/StatefulPartitionedCall2j
3pred_layer_3/bias/Regularizer/Square/ReadVariableOp3pred_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!
_user_specified_name	input_1
Î

-__inference_pred_layer_1_layer_call_fn_121803

inputs
unknown:	)
	unknown_0:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_1_layer_call_and_return_conditional_losses_120368p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ): : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)
 
_user_specified_nameinputs

»
__inference_loss_fn_8_122013R
>pred_layer_2_kernel_regularizer_square_readvariableop_resource:

identity¢5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp¶
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>pred_layer_2_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype0
&pred_layer_2/kernel/Regularizer/SquareSquare=pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%pred_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_2/kernel/Regularizer/SumSum*pred_layer_2/kernel/Regularizer/Square:y:0.pred_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_2/kernel/Regularizer/mulMul.pred_layer_2/kernel/Regularizer/mul/x:output:0,pred_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentity'pred_layer_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^pred_layer_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp
Ì
®
O__inference_batch_normalization_layer_call_and_return_conditional_losses_121735

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ê
H__inference_pred_layer_2_layer_call_and_return_conditional_losses_120397

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3pred_layer_2/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&pred_layer_2/kernel/Regularizer/SquareSquare=pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%pred_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_2/kernel/Regularizer/SumSum*pred_layer_2/kernel/Regularizer/Square:y:0.pred_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_2/kernel/Regularizer/mulMul.pred_layer_2/kernel/Regularizer/mul/x:output:0,pred_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_2/bias/Regularizer/SquareSquare;pred_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_2/bias/Regularizer/SumSum(pred_layer_2/bias/Regularizer/Square:y:0,pred_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_2/bias/Regularizer/mulMul,pred_layer_2/bias/Regularizer/mul/x:output:0*pred_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^pred_layer_2/bias/Regularizer/Square/ReadVariableOp6^pred_layer_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3pred_layer_2/bias/Regularizer/Square/ReadVariableOp3pred_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

é
H__inference_ctrl_layer_1_layer_call_and_return_conditional_losses_120263

inputs1
matmul_readvariableop_resource:	(.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype0
&ctrl_layer_1/kernel/Regularizer/SquareSquare=ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	(v
%ctrl_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_1/kernel/Regularizer/SumSum*ctrl_layer_1/kernel/Regularizer/Square:y:0.ctrl_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_1/kernel/Regularizer/mulMul.ctrl_layer_1/kernel/Regularizer/mul/x:output:0,ctrl_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_1/bias/Regularizer/SquareSquare;ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_1/bias/Regularizer/SumSum(ctrl_layer_1/bias/Regularizer/Square:y:0,ctrl_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_1/bias/Regularizer/mulMul,ctrl_layer_1/bias/Regularizer/mul/x:output:0*ctrl_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

»
__inference_loss_fn_10_122035Q
>pred_layer_3_kernel_regularizer_square_readvariableop_resource:	
identity¢5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpµ
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>pred_layer_3_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype0
&pred_layer_3/kernel/Regularizer/SquareSquare=pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%pred_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_3/kernel/Regularizer/SumSum*pred_layer_3/kernel/Regularizer/Square:y:0.pred_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_3/kernel/Regularizer/mulMul.pred_layer_3/kernel/Regularizer/mul/x:output:0,pred_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentity'pred_layer_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^pred_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp
Ñ

-__inference_pred_layer_2_layer_call_fn_121847

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_2_layer_call_and_return_conditional_losses_120397p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
²
__inference_loss_fn_3_121958K
<ctrl_layer_2_bias_regularizer_square_readvariableop_resource:	
identity¢3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp­
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp<ctrl_layer_2_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_2/bias/Regularizer/SquareSquare;ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_2/bias/Regularizer/SumSum(ctrl_layer_2/bias/Regularizer/Square:y:0,ctrl_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_2/bias/Regularizer/mulMul,ctrl_layer_2/bias/Regularizer/mul/x:output:0*ctrl_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%ctrl_layer_2/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp

é
H__inference_ctrl_layer_1_layer_call_and_return_conditional_losses_121601

inputs1
matmul_readvariableop_resource:	(.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype0
&ctrl_layer_1/kernel/Regularizer/SquareSquare=ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	(v
%ctrl_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_1/kernel/Regularizer/SumSum*ctrl_layer_1/kernel/Regularizer/Square:y:0.ctrl_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_1/kernel/Regularizer/mulMul.ctrl_layer_1/kernel/Regularizer/mul/x:output:0,ctrl_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_1/bias/Regularizer/SquareSquare;ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_1/bias/Regularizer/SumSum(ctrl_layer_1/bias/Regularizer/Square:y:0,ctrl_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_1/bias/Regularizer/mulMul,ctrl_layer_1/bias/Regularizer/mul/x:output:0*ctrl_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ð
¦
&__inference_model_layer_call_fn_121189

inputs
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	)

unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_120506o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

ê
H__inference_ctrl_layer_2_layer_call_and_return_conditional_losses_120292

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&ctrl_layer_2/kernel/Regularizer/SquareSquare=ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%ctrl_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_2/kernel/Regularizer/SumSum*ctrl_layer_2/kernel/Regularizer/Square:y:0.ctrl_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_2/kernel/Regularizer/mulMul.ctrl_layer_2/kernel/Regularizer/mul/x:output:0,ctrl_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_2/bias/Regularizer/SquareSquare;ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_2/bias/Regularizer/SumSum(ctrl_layer_2/bias/Regularizer/Square:y:0,ctrl_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_2/bias/Regularizer/mulMul,ctrl_layer_2/bias/Regularizer/mul/x:output:0*ctrl_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ

A__inference_model_layer_call_and_return_conditional_losses_121078
input_1&
ctrl_layer_1_120964:	("
ctrl_layer_1_120966:	'
ctrl_layer_2_120969:
"
ctrl_layer_2_120971:	&
ctrl_layer_3_120974:	!
ctrl_layer_3_120976:(
batch_normalization_120979:(
batch_normalization_120981:(
batch_normalization_120983:(
batch_normalization_120985:&
pred_layer_1_120989:	)"
pred_layer_1_120991:	'
pred_layer_2_120994:
"
pred_layer_2_120996:	&
pred_layer_3_120999:	!
pred_layer_3_121001:
identity

identity_1¢+batch_normalization/StatefulPartitionedCall¢$ctrl_layer_1/StatefulPartitionedCall¢3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp¢$ctrl_layer_2/StatefulPartitionedCall¢3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp¢$ctrl_layer_3/StatefulPartitionedCall¢3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_1/StatefulPartitionedCall¢3pred_layer_1/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_2/StatefulPartitionedCall¢3pred_layer_2/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_3/StatefulPartitionedCall¢3pred_layer_3/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp
$ctrl_layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_1ctrl_layer_1_120964ctrl_layer_1_120966*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_1_layer_call_and_return_conditional_losses_120263¨
$ctrl_layer_2/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_1/StatefulPartitionedCall:output:0ctrl_layer_2_120969ctrl_layer_2_120971*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_2_layer_call_and_return_conditional_losses_120292§
$ctrl_layer_3/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_2/StatefulPartitionedCall:output:0ctrl_layer_3_120974ctrl_layer_3_120976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_3_layer_call_and_return_conditional_losses_120321ý
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_3/StatefulPartitionedCall:output:0batch_normalization_120979batch_normalization_120981batch_normalization_120983batch_normalization_120985*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_120222ö
concatenate/PartitionedCallPartitionedCallinput_14batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_120343
$pred_layer_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0pred_layer_1_120989pred_layer_1_120991*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_1_layer_call_and_return_conditional_losses_120368¨
$pred_layer_2/StatefulPartitionedCallStatefulPartitionedCall-pred_layer_1/StatefulPartitionedCall:output:0pred_layer_2_120994pred_layer_2_120996*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_2_layer_call_and_return_conditional_losses_120397§
$pred_layer_3/StatefulPartitionedCallStatefulPartitionedCall-pred_layer_2/StatefulPartitionedCall:output:0pred_layer_3_120999pred_layer_3_121001*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_3_layer_call_and_return_conditional_losses_120426
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_1_120964*
_output_shapes
:	(*
dtype0
&ctrl_layer_1/kernel/Regularizer/SquareSquare=ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	(v
%ctrl_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_1/kernel/Regularizer/SumSum*ctrl_layer_1/kernel/Regularizer/Square:y:0.ctrl_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_1/kernel/Regularizer/mulMul.ctrl_layer_1/kernel/Regularizer/mul/x:output:0,ctrl_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_1_120966*
_output_shapes	
:*
dtype0
$ctrl_layer_1/bias/Regularizer/SquareSquare;ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_1/bias/Regularizer/SumSum(ctrl_layer_1/bias/Regularizer/Square:y:0,ctrl_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_1/bias/Regularizer/mulMul,ctrl_layer_1/bias/Regularizer/mul/x:output:0*ctrl_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_2_120969* 
_output_shapes
:
*
dtype0
&ctrl_layer_2/kernel/Regularizer/SquareSquare=ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%ctrl_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_2/kernel/Regularizer/SumSum*ctrl_layer_2/kernel/Regularizer/Square:y:0.ctrl_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_2/kernel/Regularizer/mulMul.ctrl_layer_2/kernel/Regularizer/mul/x:output:0,ctrl_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_2_120971*
_output_shapes	
:*
dtype0
$ctrl_layer_2/bias/Regularizer/SquareSquare;ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_2/bias/Regularizer/SumSum(ctrl_layer_2/bias/Regularizer/Square:y:0,ctrl_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_2/bias/Regularizer/mulMul,ctrl_layer_2/bias/Regularizer/mul/x:output:0*ctrl_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_3_120974*
_output_shapes
:	*
dtype0
&ctrl_layer_3/kernel/Regularizer/SquareSquare=ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%ctrl_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_3/kernel/Regularizer/SumSum*ctrl_layer_3/kernel/Regularizer/Square:y:0.ctrl_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_3/kernel/Regularizer/mulMul.ctrl_layer_3/kernel/Regularizer/mul/x:output:0,ctrl_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_3_120976*
_output_shapes
:*
dtype0
$ctrl_layer_3/bias/Regularizer/SquareSquare;ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#ctrl_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_3/bias/Regularizer/SumSum(ctrl_layer_3/bias/Regularizer/Square:y:0,ctrl_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_3/bias/Regularizer/mulMul,ctrl_layer_3/bias/Regularizer/mul/x:output:0*ctrl_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_1_120989*
_output_shapes
:	)*
dtype0
&pred_layer_1/kernel/Regularizer/SquareSquare=pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	)v
%pred_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_1/kernel/Regularizer/SumSum*pred_layer_1/kernel/Regularizer/Square:y:0.pred_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_1/kernel/Regularizer/mulMul.pred_layer_1/kernel/Regularizer/mul/x:output:0,pred_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_1_120991*
_output_shapes	
:*
dtype0
$pred_layer_1/bias/Regularizer/SquareSquare;pred_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_1/bias/Regularizer/SumSum(pred_layer_1/bias/Regularizer/Square:y:0,pred_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_1/bias/Regularizer/mulMul,pred_layer_1/bias/Regularizer/mul/x:output:0*pred_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_2_120994* 
_output_shapes
:
*
dtype0
&pred_layer_2/kernel/Regularizer/SquareSquare=pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%pred_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_2/kernel/Regularizer/SumSum*pred_layer_2/kernel/Regularizer/Square:y:0.pred_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_2/kernel/Regularizer/mulMul.pred_layer_2/kernel/Regularizer/mul/x:output:0,pred_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_2_120996*
_output_shapes	
:*
dtype0
$pred_layer_2/bias/Regularizer/SquareSquare;pred_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_2/bias/Regularizer/SumSum(pred_layer_2/bias/Regularizer/Square:y:0,pred_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_2/bias/Regularizer/mulMul,pred_layer_2/bias/Regularizer/mul/x:output:0*pred_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_3_120999*
_output_shapes
:	*
dtype0
&pred_layer_3/kernel/Regularizer/SquareSquare=pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%pred_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_3/kernel/Regularizer/SumSum*pred_layer_3/kernel/Regularizer/Square:y:0.pred_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_3/kernel/Regularizer/mulMul.pred_layer_3/kernel/Regularizer/mul/x:output:0,pred_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_3_121001*
_output_shapes
:*
dtype0
$pred_layer_3/bias/Regularizer/SquareSquare;pred_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#pred_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_3/bias/Regularizer/SumSum(pred_layer_3/bias/Regularizer/Square:y:0,pred_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_3/bias/Regularizer/mulMul,pred_layer_3/bias/Regularizer/mul/x:output:0*pred_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-pred_layer_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~

Identity_1Identity-ctrl_layer_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOp,^batch_normalization/StatefulPartitionedCall%^ctrl_layer_1/StatefulPartitionedCall4^ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp%^ctrl_layer_2/StatefulPartitionedCall4^ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp%^ctrl_layer_3/StatefulPartitionedCall4^ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_1/StatefulPartitionedCall4^pred_layer_1/bias/Regularizer/Square/ReadVariableOp6^pred_layer_1/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_2/StatefulPartitionedCall4^pred_layer_2/bias/Regularizer/Square/ReadVariableOp6^pred_layer_2/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_3/StatefulPartitionedCall4^pred_layer_3/bias/Regularizer/Square/ReadVariableOp6^pred_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$ctrl_layer_1/StatefulPartitionedCall$ctrl_layer_1/StatefulPartitionedCall2j
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp2L
$ctrl_layer_2/StatefulPartitionedCall$ctrl_layer_2/StatefulPartitionedCall2j
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$ctrl_layer_3/StatefulPartitionedCall$ctrl_layer_3/StatefulPartitionedCall2j
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_1/StatefulPartitionedCall$pred_layer_1/StatefulPartitionedCall2j
3pred_layer_1/bias/Regularizer/Square/ReadVariableOp3pred_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_2/StatefulPartitionedCall$pred_layer_2/StatefulPartitionedCall2j
3pred_layer_2/bias/Regularizer/Square/ReadVariableOp3pred_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_3/StatefulPartitionedCall$pred_layer_3/StatefulPartitionedCall2j
3pred_layer_3/bias/Regularizer/Square/ReadVariableOp3pred_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!
_user_specified_name	input_1
¤
Ï
4__inference_batch_normalization_layer_call_fn_121702

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_120175o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í7
Ó

__inference__traced_save_122148
file_prefix2
.savev2_ctrl_layer_1_kernel_read_readvariableop0
,savev2_ctrl_layer_1_bias_read_readvariableop2
.savev2_ctrl_layer_2_kernel_read_readvariableop0
,savev2_ctrl_layer_2_bias_read_readvariableop2
.savev2_ctrl_layer_3_kernel_read_readvariableop0
,savev2_ctrl_layer_3_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop2
.savev2_pred_layer_1_kernel_read_readvariableop0
,savev2_pred_layer_1_bias_read_readvariableop2
.savev2_pred_layer_2_kernel_read_readvariableop0
,savev2_pred_layer_2_bias_read_readvariableop2
.savev2_pred_layer_3_kernel_read_readvariableop0
,savev2_pred_layer_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¥
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î
valueÄBÁB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH£
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B É

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_ctrl_layer_1_kernel_read_readvariableop,savev2_ctrl_layer_1_bias_read_readvariableop.savev2_ctrl_layer_2_kernel_read_readvariableop,savev2_ctrl_layer_2_bias_read_readvariableop.savev2_ctrl_layer_3_kernel_read_readvariableop,savev2_ctrl_layer_3_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop.savev2_pred_layer_1_kernel_read_readvariableop,savev2_pred_layer_1_bias_read_readvariableop.savev2_pred_layer_2_kernel_read_readvariableop,savev2_pred_layer_2_bias_read_readvariableop.savev2_pred_layer_3_kernel_read_readvariableop,savev2_pred_layer_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*±
_input_shapes
: :	(::
::	::::::	)::
::	:: : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	(:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::%!

_output_shapes
:	):!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

º
__inference_loss_fn_6_121991Q
>pred_layer_1_kernel_regularizer_square_readvariableop_resource:	)
identity¢5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpµ
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>pred_layer_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	)*
dtype0
&pred_layer_1/kernel/Regularizer/SquareSquare=pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	)v
%pred_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_1/kernel/Regularizer/SumSum*pred_layer_1/kernel/Regularizer/Square:y:0.pred_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_1/kernel/Regularizer/mulMul.pred_layer_1/kernel/Regularizer/mul/x:output:0,pred_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentity'pred_layer_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^pred_layer_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp
Ë
²
__inference_loss_fn_7_122002K
<pred_layer_1_bias_regularizer_square_readvariableop_resource:	
identity¢3pred_layer_1/bias/Regularizer/Square/ReadVariableOp­
3pred_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp<pred_layer_1_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_1/bias/Regularizer/SquareSquare;pred_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_1/bias/Regularizer/SumSum(pred_layer_1/bias/Regularizer/Square:y:0,pred_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_1/bias/Regularizer/mulMul,pred_layer_1/bias/Regularizer/mul/x:output:0*pred_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%pred_layer_1/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^pred_layer_1/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3pred_layer_1/bias/Regularizer/Square/ReadVariableOp3pred_layer_1/bias/Regularizer/Square/ReadVariableOp
Ì
®
O__inference_batch_normalization_layer_call_and_return_conditional_losses_120175

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
q
G__inference_concatenate_layer_call_and_return_conditional_losses_120343

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

A__inference_model_layer_call_and_return_conditional_losses_120768

inputs&
ctrl_layer_1_120654:	("
ctrl_layer_1_120656:	'
ctrl_layer_2_120659:
"
ctrl_layer_2_120661:	&
ctrl_layer_3_120664:	!
ctrl_layer_3_120666:(
batch_normalization_120669:(
batch_normalization_120671:(
batch_normalization_120673:(
batch_normalization_120675:&
pred_layer_1_120679:	)"
pred_layer_1_120681:	'
pred_layer_2_120684:
"
pred_layer_2_120686:	&
pred_layer_3_120689:	!
pred_layer_3_120691:
identity

identity_1¢+batch_normalization/StatefulPartitionedCall¢$ctrl_layer_1/StatefulPartitionedCall¢3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp¢$ctrl_layer_2/StatefulPartitionedCall¢3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp¢$ctrl_layer_3/StatefulPartitionedCall¢3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_1/StatefulPartitionedCall¢3pred_layer_1/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_2/StatefulPartitionedCall¢3pred_layer_2/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp¢$pred_layer_3/StatefulPartitionedCall¢3pred_layer_3/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp
$ctrl_layer_1/StatefulPartitionedCallStatefulPartitionedCallinputsctrl_layer_1_120654ctrl_layer_1_120656*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_1_layer_call_and_return_conditional_losses_120263¨
$ctrl_layer_2/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_1/StatefulPartitionedCall:output:0ctrl_layer_2_120659ctrl_layer_2_120661*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_2_layer_call_and_return_conditional_losses_120292§
$ctrl_layer_3/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_2/StatefulPartitionedCall:output:0ctrl_layer_3_120664ctrl_layer_3_120666*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_3_layer_call_and_return_conditional_losses_120321ý
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall-ctrl_layer_3/StatefulPartitionedCall:output:0batch_normalization_120669batch_normalization_120671batch_normalization_120673batch_normalization_120675*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_120222õ
concatenate/PartitionedCallPartitionedCallinputs4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_120343
$pred_layer_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0pred_layer_1_120679pred_layer_1_120681*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_1_layer_call_and_return_conditional_losses_120368¨
$pred_layer_2/StatefulPartitionedCallStatefulPartitionedCall-pred_layer_1/StatefulPartitionedCall:output:0pred_layer_2_120684pred_layer_2_120686*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_2_layer_call_and_return_conditional_losses_120397§
$pred_layer_3/StatefulPartitionedCallStatefulPartitionedCall-pred_layer_2/StatefulPartitionedCall:output:0pred_layer_3_120689pred_layer_3_120691*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_3_layer_call_and_return_conditional_losses_120426
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_1_120654*
_output_shapes
:	(*
dtype0
&ctrl_layer_1/kernel/Regularizer/SquareSquare=ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	(v
%ctrl_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_1/kernel/Regularizer/SumSum*ctrl_layer_1/kernel/Regularizer/Square:y:0.ctrl_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_1/kernel/Regularizer/mulMul.ctrl_layer_1/kernel/Regularizer/mul/x:output:0,ctrl_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_1_120656*
_output_shapes	
:*
dtype0
$ctrl_layer_1/bias/Regularizer/SquareSquare;ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_1/bias/Regularizer/SumSum(ctrl_layer_1/bias/Regularizer/Square:y:0,ctrl_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_1/bias/Regularizer/mulMul,ctrl_layer_1/bias/Regularizer/mul/x:output:0*ctrl_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_2_120659* 
_output_shapes
:
*
dtype0
&ctrl_layer_2/kernel/Regularizer/SquareSquare=ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%ctrl_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_2/kernel/Regularizer/SumSum*ctrl_layer_2/kernel/Regularizer/Square:y:0.ctrl_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_2/kernel/Regularizer/mulMul.ctrl_layer_2/kernel/Regularizer/mul/x:output:0,ctrl_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_2_120661*
_output_shapes	
:*
dtype0
$ctrl_layer_2/bias/Regularizer/SquareSquare;ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_2/bias/Regularizer/SumSum(ctrl_layer_2/bias/Regularizer/Square:y:0,ctrl_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_2/bias/Regularizer/mulMul,ctrl_layer_2/bias/Regularizer/mul/x:output:0*ctrl_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_3_120664*
_output_shapes
:	*
dtype0
&ctrl_layer_3/kernel/Regularizer/SquareSquare=ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%ctrl_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_3/kernel/Regularizer/SumSum*ctrl_layer_3/kernel/Regularizer/Square:y:0.ctrl_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_3/kernel/Regularizer/mulMul.ctrl_layer_3/kernel/Regularizer/mul/x:output:0,ctrl_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpctrl_layer_3_120666*
_output_shapes
:*
dtype0
$ctrl_layer_3/bias/Regularizer/SquareSquare;ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#ctrl_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_3/bias/Regularizer/SumSum(ctrl_layer_3/bias/Regularizer/Square:y:0,ctrl_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_3/bias/Regularizer/mulMul,ctrl_layer_3/bias/Regularizer/mul/x:output:0*ctrl_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_1_120679*
_output_shapes
:	)*
dtype0
&pred_layer_1/kernel/Regularizer/SquareSquare=pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	)v
%pred_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_1/kernel/Regularizer/SumSum*pred_layer_1/kernel/Regularizer/Square:y:0.pred_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_1/kernel/Regularizer/mulMul.pred_layer_1/kernel/Regularizer/mul/x:output:0,pred_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_1_120681*
_output_shapes	
:*
dtype0
$pred_layer_1/bias/Regularizer/SquareSquare;pred_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_1/bias/Regularizer/SumSum(pred_layer_1/bias/Regularizer/Square:y:0,pred_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_1/bias/Regularizer/mulMul,pred_layer_1/bias/Regularizer/mul/x:output:0*pred_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_2_120684* 
_output_shapes
:
*
dtype0
&pred_layer_2/kernel/Regularizer/SquareSquare=pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%pred_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_2/kernel/Regularizer/SumSum*pred_layer_2/kernel/Regularizer/Square:y:0.pred_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_2/kernel/Regularizer/mulMul.pred_layer_2/kernel/Regularizer/mul/x:output:0,pred_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_2_120686*
_output_shapes	
:*
dtype0
$pred_layer_2/bias/Regularizer/SquareSquare;pred_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_2/bias/Regularizer/SumSum(pred_layer_2/bias/Regularizer/Square:y:0,pred_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_2/bias/Regularizer/mulMul,pred_layer_2/bias/Regularizer/mul/x:output:0*pred_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_3_120689*
_output_shapes
:	*
dtype0
&pred_layer_3/kernel/Regularizer/SquareSquare=pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%pred_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_3/kernel/Regularizer/SumSum*pred_layer_3/kernel/Regularizer/Square:y:0.pred_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_3/kernel/Regularizer/mulMul.pred_layer_3/kernel/Regularizer/mul/x:output:0,pred_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOppred_layer_3_120691*
_output_shapes
:*
dtype0
$pred_layer_3/bias/Regularizer/SquareSquare;pred_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#pred_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_3/bias/Regularizer/SumSum(pred_layer_3/bias/Regularizer/Square:y:0,pred_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_3/bias/Regularizer/mulMul,pred_layer_3/bias/Regularizer/mul/x:output:0*pred_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: |
IdentityIdentity-pred_layer_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~

Identity_1Identity-ctrl_layer_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
NoOpNoOp,^batch_normalization/StatefulPartitionedCall%^ctrl_layer_1/StatefulPartitionedCall4^ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp%^ctrl_layer_2/StatefulPartitionedCall4^ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp%^ctrl_layer_3/StatefulPartitionedCall4^ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_1/StatefulPartitionedCall4^pred_layer_1/bias/Regularizer/Square/ReadVariableOp6^pred_layer_1/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_2/StatefulPartitionedCall4^pred_layer_2/bias/Regularizer/Square/ReadVariableOp6^pred_layer_2/kernel/Regularizer/Square/ReadVariableOp%^pred_layer_3/StatefulPartitionedCall4^pred_layer_3/bias/Regularizer/Square/ReadVariableOp6^pred_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2L
$ctrl_layer_1/StatefulPartitionedCall$ctrl_layer_1/StatefulPartitionedCall2j
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp2L
$ctrl_layer_2/StatefulPartitionedCall$ctrl_layer_2/StatefulPartitionedCall2j
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$ctrl_layer_3/StatefulPartitionedCall$ctrl_layer_3/StatefulPartitionedCall2j
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_1/StatefulPartitionedCall$pred_layer_1/StatefulPartitionedCall2j
3pred_layer_1/bias/Regularizer/Square/ReadVariableOp3pred_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_2/StatefulPartitionedCall$pred_layer_2/StatefulPartitionedCall2j
3pred_layer_2/bias/Regularizer/Square/ReadVariableOp3pred_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp2L
$pred_layer_3/StatefulPartitionedCall$pred_layer_3/StatefulPartitionedCall2j
3pred_layer_3/bias/Regularizer/Square/ReadVariableOp3pred_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
%
è
O__inference_batch_normalization_layer_call_and_return_conditional_losses_121769

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ê
H__inference_ctrl_layer_2_layer_call_and_return_conditional_losses_121645

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&ctrl_layer_2/kernel/Regularizer/SquareSquare=ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%ctrl_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_2/kernel/Regularizer/SumSum*ctrl_layer_2/kernel/Regularizer/Square:y:0.ctrl_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_2/kernel/Regularizer/mulMul.ctrl_layer_2/kernel/Regularizer/mul/x:output:0,ctrl_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_2/bias/Regularizer/SquareSquare;ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_2/bias/Regularizer/SumSum(ctrl_layer_2/bias/Regularizer/Square:y:0,ctrl_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_2/bias/Regularizer/mulMul,ctrl_layer_2/bias/Regularizer/mul/x:output:0*ctrl_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

è
H__inference_pred_layer_3_layer_call_and_return_conditional_losses_121914

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3pred_layer_3/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&pred_layer_3/kernel/Regularizer/SquareSquare=pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%pred_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_3/kernel/Regularizer/SumSum*pred_layer_3/kernel/Regularizer/Square:y:0.pred_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_3/kernel/Regularizer/mulMul.pred_layer_3/kernel/Regularizer/mul/x:output:0,pred_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$pred_layer_3/bias/Regularizer/SquareSquare;pred_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#pred_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_3/bias/Regularizer/SumSum(pred_layer_3/bias/Regularizer/Square:y:0,pred_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_3/bias/Regularizer/mulMul,pred_layer_3/bias/Regularizer/mul/x:output:0*pred_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^pred_layer_3/bias/Regularizer/Square/ReadVariableOp6^pred_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3pred_layer_3/bias/Regularizer/Square/ReadVariableOp3pred_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
âÓ
ö
A__inference_model_layer_call_and_return_conditional_losses_121516

inputs>
+ctrl_layer_1_matmul_readvariableop_resource:	(;
,ctrl_layer_1_biasadd_readvariableop_resource:	?
+ctrl_layer_2_matmul_readvariableop_resource:
;
,ctrl_layer_2_biasadd_readvariableop_resource:	>
+ctrl_layer_3_matmul_readvariableop_resource:	:
,ctrl_layer_3_biasadd_readvariableop_resource:I
;batch_normalization_assignmovingavg_readvariableop_resource:K
=batch_normalization_assignmovingavg_1_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:C
5batch_normalization_batchnorm_readvariableop_resource:>
+pred_layer_1_matmul_readvariableop_resource:	);
,pred_layer_1_biasadd_readvariableop_resource:	?
+pred_layer_2_matmul_readvariableop_resource:
;
,pred_layer_2_biasadd_readvariableop_resource:	>
+pred_layer_3_matmul_readvariableop_resource:	:
,pred_layer_3_biasadd_readvariableop_resource:
identity

identity_1¢#batch_normalization/AssignMovingAvg¢2batch_normalization/AssignMovingAvg/ReadVariableOp¢%batch_normalization/AssignMovingAvg_1¢4batch_normalization/AssignMovingAvg_1/ReadVariableOp¢,batch_normalization/batchnorm/ReadVariableOp¢0batch_normalization/batchnorm/mul/ReadVariableOp¢#ctrl_layer_1/BiasAdd/ReadVariableOp¢"ctrl_layer_1/MatMul/ReadVariableOp¢3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp¢#ctrl_layer_2/BiasAdd/ReadVariableOp¢"ctrl_layer_2/MatMul/ReadVariableOp¢3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp¢#ctrl_layer_3/BiasAdd/ReadVariableOp¢"ctrl_layer_3/MatMul/ReadVariableOp¢3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp¢#pred_layer_1/BiasAdd/ReadVariableOp¢"pred_layer_1/MatMul/ReadVariableOp¢3pred_layer_1/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp¢#pred_layer_2/BiasAdd/ReadVariableOp¢"pred_layer_2/MatMul/ReadVariableOp¢3pred_layer_2/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp¢#pred_layer_3/BiasAdd/ReadVariableOp¢"pred_layer_3/MatMul/ReadVariableOp¢3pred_layer_3/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp
"ctrl_layer_1/MatMul/ReadVariableOpReadVariableOp+ctrl_layer_1_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype0
ctrl_layer_1/MatMulMatMulinputs*ctrl_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#ctrl_layer_1/BiasAdd/ReadVariableOpReadVariableOp,ctrl_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
ctrl_layer_1/BiasAddBiasAddctrl_layer_1/MatMul:product:0+ctrl_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
ctrl_layer_1/ReluReluctrl_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"ctrl_layer_2/MatMul/ReadVariableOpReadVariableOp+ctrl_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
ctrl_layer_2/MatMulMatMulctrl_layer_1/Relu:activations:0*ctrl_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#ctrl_layer_2/BiasAdd/ReadVariableOpReadVariableOp,ctrl_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
ctrl_layer_2/BiasAddBiasAddctrl_layer_2/MatMul:product:0+ctrl_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
ctrl_layer_2/ReluReluctrl_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"ctrl_layer_3/MatMul/ReadVariableOpReadVariableOp+ctrl_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
ctrl_layer_3/MatMulMatMulctrl_layer_2/Relu:activations:0*ctrl_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#ctrl_layer_3/BiasAdd/ReadVariableOpReadVariableOp,ctrl_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
ctrl_layer_3/BiasAddBiasAddctrl_layer_3/MatMul:product:0+ctrl_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ctrl_layer_3/ReluReluctrl_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: À
 batch_normalization/moments/meanMeanctrl_layer_3/Relu:activations:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:È
-batch_normalization/moments/SquaredDifferenceSquaredDifferencectrl_layer_3/Relu:activations:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ú
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<ª
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0½
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:´
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ü
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<®
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Ã
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:º
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:­
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:¦
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0°
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
#batch_normalization/batchnorm/mul_1Mulctrl_layer_3/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0¬
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:®
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¬
concatenate/concatConcatV2inputs'batch_normalization/batchnorm/add_1:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)
"pred_layer_1/MatMul/ReadVariableOpReadVariableOp+pred_layer_1_matmul_readvariableop_resource*
_output_shapes
:	)*
dtype0
pred_layer_1/MatMulMatMulconcatenate/concat:output:0*pred_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#pred_layer_1/BiasAdd/ReadVariableOpReadVariableOp,pred_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
pred_layer_1/BiasAddBiasAddpred_layer_1/MatMul:product:0+pred_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
pred_layer_1/ReluRelupred_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"pred_layer_2/MatMul/ReadVariableOpReadVariableOp+pred_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
pred_layer_2/MatMulMatMulpred_layer_1/Relu:activations:0*pred_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#pred_layer_2/BiasAdd/ReadVariableOpReadVariableOp,pred_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
pred_layer_2/BiasAddBiasAddpred_layer_2/MatMul:product:0+pred_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
pred_layer_2/ReluRelupred_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"pred_layer_3/MatMul/ReadVariableOpReadVariableOp+pred_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
pred_layer_3/MatMulMatMulpred_layer_2/Relu:activations:0*pred_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#pred_layer_3/BiasAdd/ReadVariableOpReadVariableOp,pred_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
pred_layer_3/BiasAddBiasAddpred_layer_3/MatMul:product:0+pred_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
pred_layer_3/ReluRelupred_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+ctrl_layer_1_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype0
&ctrl_layer_1/kernel/Regularizer/SquareSquare=ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	(v
%ctrl_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_1/kernel/Regularizer/SumSum*ctrl_layer_1/kernel/Regularizer/Square:y:0.ctrl_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_1/kernel/Regularizer/mulMul.ctrl_layer_1/kernel/Regularizer/mul/x:output:0,ctrl_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp,ctrl_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_1/bias/Regularizer/SquareSquare;ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_1/bias/Regularizer/SumSum(ctrl_layer_1/bias/Regularizer/Square:y:0,ctrl_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_1/bias/Regularizer/mulMul,ctrl_layer_1/bias/Regularizer/mul/x:output:0*ctrl_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+ctrl_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&ctrl_layer_2/kernel/Regularizer/SquareSquare=ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%ctrl_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_2/kernel/Regularizer/SumSum*ctrl_layer_2/kernel/Regularizer/Square:y:0.ctrl_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_2/kernel/Regularizer/mulMul.ctrl_layer_2/kernel/Regularizer/mul/x:output:0,ctrl_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp,ctrl_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_2/bias/Regularizer/SquareSquare;ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_2/bias/Regularizer/SumSum(ctrl_layer_2/bias/Regularizer/Square:y:0,ctrl_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_2/bias/Regularizer/mulMul,ctrl_layer_2/bias/Regularizer/mul/x:output:0*ctrl_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+ctrl_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&ctrl_layer_3/kernel/Regularizer/SquareSquare=ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%ctrl_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_3/kernel/Regularizer/SumSum*ctrl_layer_3/kernel/Regularizer/Square:y:0.ctrl_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_3/kernel/Regularizer/mulMul.ctrl_layer_3/kernel/Regularizer/mul/x:output:0,ctrl_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp,ctrl_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$ctrl_layer_3/bias/Regularizer/SquareSquare;ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#ctrl_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_3/bias/Regularizer/SumSum(ctrl_layer_3/bias/Regularizer/Square:y:0,ctrl_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_3/bias/Regularizer/mulMul,ctrl_layer_3/bias/Regularizer/mul/x:output:0*ctrl_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+pred_layer_1_matmul_readvariableop_resource*
_output_shapes
:	)*
dtype0
&pred_layer_1/kernel/Regularizer/SquareSquare=pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	)v
%pred_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_1/kernel/Regularizer/SumSum*pred_layer_1/kernel/Regularizer/Square:y:0.pred_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_1/kernel/Regularizer/mulMul.pred_layer_1/kernel/Regularizer/mul/x:output:0,pred_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp,pred_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_1/bias/Regularizer/SquareSquare;pred_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_1/bias/Regularizer/SumSum(pred_layer_1/bias/Regularizer/Square:y:0,pred_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_1/bias/Regularizer/mulMul,pred_layer_1/bias/Regularizer/mul/x:output:0*pred_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+pred_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&pred_layer_2/kernel/Regularizer/SquareSquare=pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%pred_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_2/kernel/Regularizer/SumSum*pred_layer_2/kernel/Regularizer/Square:y:0.pred_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_2/kernel/Regularizer/mulMul.pred_layer_2/kernel/Regularizer/mul/x:output:0,pred_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp,pred_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_2/bias/Regularizer/SquareSquare;pred_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_2/bias/Regularizer/SumSum(pred_layer_2/bias/Regularizer/Square:y:0,pred_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_2/bias/Regularizer/mulMul,pred_layer_2/bias/Regularizer/mul/x:output:0*pred_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+pred_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&pred_layer_3/kernel/Regularizer/SquareSquare=pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%pred_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_3/kernel/Regularizer/SumSum*pred_layer_3/kernel/Regularizer/Square:y:0.pred_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_3/kernel/Regularizer/mulMul.pred_layer_3/kernel/Regularizer/mul/x:output:0,pred_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp,pred_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$pred_layer_3/bias/Regularizer/SquareSquare;pred_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#pred_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_3/bias/Regularizer/SumSum(pred_layer_3/bias/Regularizer/Square:y:0,pred_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_3/bias/Regularizer/mulMul,pred_layer_3/bias/Regularizer/mul/x:output:0*pred_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentitypred_layer_3/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp

Identity_1Identityctrl_layer_3/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp$^ctrl_layer_1/BiasAdd/ReadVariableOp#^ctrl_layer_1/MatMul/ReadVariableOp4^ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp$^ctrl_layer_2/BiasAdd/ReadVariableOp#^ctrl_layer_2/MatMul/ReadVariableOp4^ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp$^ctrl_layer_3/BiasAdd/ReadVariableOp#^ctrl_layer_3/MatMul/ReadVariableOp4^ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp$^pred_layer_1/BiasAdd/ReadVariableOp#^pred_layer_1/MatMul/ReadVariableOp4^pred_layer_1/bias/Regularizer/Square/ReadVariableOp6^pred_layer_1/kernel/Regularizer/Square/ReadVariableOp$^pred_layer_2/BiasAdd/ReadVariableOp#^pred_layer_2/MatMul/ReadVariableOp4^pred_layer_2/bias/Regularizer/Square/ReadVariableOp6^pred_layer_2/kernel/Regularizer/Square/ReadVariableOp$^pred_layer_3/BiasAdd/ReadVariableOp#^pred_layer_3/MatMul/ReadVariableOp4^pred_layer_3/bias/Regularizer/Square/ReadVariableOp6^pred_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2J
#ctrl_layer_1/BiasAdd/ReadVariableOp#ctrl_layer_1/BiasAdd/ReadVariableOp2H
"ctrl_layer_1/MatMul/ReadVariableOp"ctrl_layer_1/MatMul/ReadVariableOp2j
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp2J
#ctrl_layer_2/BiasAdd/ReadVariableOp#ctrl_layer_2/BiasAdd/ReadVariableOp2H
"ctrl_layer_2/MatMul/ReadVariableOp"ctrl_layer_2/MatMul/ReadVariableOp2j
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp2J
#ctrl_layer_3/BiasAdd/ReadVariableOp#ctrl_layer_3/BiasAdd/ReadVariableOp2H
"ctrl_layer_3/MatMul/ReadVariableOp"ctrl_layer_3/MatMul/ReadVariableOp2j
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp2J
#pred_layer_1/BiasAdd/ReadVariableOp#pred_layer_1/BiasAdd/ReadVariableOp2H
"pred_layer_1/MatMul/ReadVariableOp"pred_layer_1/MatMul/ReadVariableOp2j
3pred_layer_1/bias/Regularizer/Square/ReadVariableOp3pred_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp2J
#pred_layer_2/BiasAdd/ReadVariableOp#pred_layer_2/BiasAdd/ReadVariableOp2H
"pred_layer_2/MatMul/ReadVariableOp"pred_layer_2/MatMul/ReadVariableOp2j
3pred_layer_2/bias/Regularizer/Square/ReadVariableOp3pred_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp2J
#pred_layer_3/BiasAdd/ReadVariableOp#pred_layer_3/BiasAdd/ReadVariableOp2H
"pred_layer_3/MatMul/ReadVariableOp"pred_layer_3/MatMul/ReadVariableOp2j
3pred_layer_3/bias/Regularizer/Square/ReadVariableOp3pred_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ãº

A__inference_model_layer_call_and_return_conditional_losses_121365

inputs>
+ctrl_layer_1_matmul_readvariableop_resource:	(;
,ctrl_layer_1_biasadd_readvariableop_resource:	?
+ctrl_layer_2_matmul_readvariableop_resource:
;
,ctrl_layer_2_biasadd_readvariableop_resource:	>
+ctrl_layer_3_matmul_readvariableop_resource:	:
,ctrl_layer_3_biasadd_readvariableop_resource:C
5batch_normalization_batchnorm_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:E
7batch_normalization_batchnorm_readvariableop_1_resource:E
7batch_normalization_batchnorm_readvariableop_2_resource:>
+pred_layer_1_matmul_readvariableop_resource:	);
,pred_layer_1_biasadd_readvariableop_resource:	?
+pred_layer_2_matmul_readvariableop_resource:
;
,pred_layer_2_biasadd_readvariableop_resource:	>
+pred_layer_3_matmul_readvariableop_resource:	:
,pred_layer_3_biasadd_readvariableop_resource:
identity

identity_1¢,batch_normalization/batchnorm/ReadVariableOp¢.batch_normalization/batchnorm/ReadVariableOp_1¢.batch_normalization/batchnorm/ReadVariableOp_2¢0batch_normalization/batchnorm/mul/ReadVariableOp¢#ctrl_layer_1/BiasAdd/ReadVariableOp¢"ctrl_layer_1/MatMul/ReadVariableOp¢3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp¢#ctrl_layer_2/BiasAdd/ReadVariableOp¢"ctrl_layer_2/MatMul/ReadVariableOp¢3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp¢#ctrl_layer_3/BiasAdd/ReadVariableOp¢"ctrl_layer_3/MatMul/ReadVariableOp¢3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp¢#pred_layer_1/BiasAdd/ReadVariableOp¢"pred_layer_1/MatMul/ReadVariableOp¢3pred_layer_1/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp¢#pred_layer_2/BiasAdd/ReadVariableOp¢"pred_layer_2/MatMul/ReadVariableOp¢3pred_layer_2/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp¢#pred_layer_3/BiasAdd/ReadVariableOp¢"pred_layer_3/MatMul/ReadVariableOp¢3pred_layer_3/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp
"ctrl_layer_1/MatMul/ReadVariableOpReadVariableOp+ctrl_layer_1_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype0
ctrl_layer_1/MatMulMatMulinputs*ctrl_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#ctrl_layer_1/BiasAdd/ReadVariableOpReadVariableOp,ctrl_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
ctrl_layer_1/BiasAddBiasAddctrl_layer_1/MatMul:product:0+ctrl_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
ctrl_layer_1/ReluReluctrl_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"ctrl_layer_2/MatMul/ReadVariableOpReadVariableOp+ctrl_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
ctrl_layer_2/MatMulMatMulctrl_layer_1/Relu:activations:0*ctrl_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#ctrl_layer_2/BiasAdd/ReadVariableOpReadVariableOp,ctrl_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
ctrl_layer_2/BiasAddBiasAddctrl_layer_2/MatMul:product:0+ctrl_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
ctrl_layer_2/ReluReluctrl_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"ctrl_layer_3/MatMul/ReadVariableOpReadVariableOp+ctrl_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
ctrl_layer_3/MatMulMatMulctrl_layer_2/Relu:activations:0*ctrl_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#ctrl_layer_3/BiasAdd/ReadVariableOpReadVariableOp,ctrl_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
ctrl_layer_3/BiasAddBiasAddctrl_layer_3/MatMul:product:0+ctrl_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
ctrl_layer_3/ReluReluctrl_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:¦
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0°
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¤
#batch_normalization/batchnorm/mul_1Mulctrl_layer_3/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0®
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:¢
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0®
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:®
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¬
concatenate/concatConcatV2inputs'batch_normalization/batchnorm/add_1:z:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)
"pred_layer_1/MatMul/ReadVariableOpReadVariableOp+pred_layer_1_matmul_readvariableop_resource*
_output_shapes
:	)*
dtype0
pred_layer_1/MatMulMatMulconcatenate/concat:output:0*pred_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#pred_layer_1/BiasAdd/ReadVariableOpReadVariableOp,pred_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
pred_layer_1/BiasAddBiasAddpred_layer_1/MatMul:product:0+pred_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
pred_layer_1/ReluRelupred_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"pred_layer_2/MatMul/ReadVariableOpReadVariableOp+pred_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
pred_layer_2/MatMulMatMulpred_layer_1/Relu:activations:0*pred_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#pred_layer_2/BiasAdd/ReadVariableOpReadVariableOp,pred_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
pred_layer_2/BiasAddBiasAddpred_layer_2/MatMul:product:0+pred_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
pred_layer_2/ReluRelupred_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"pred_layer_3/MatMul/ReadVariableOpReadVariableOp+pred_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
pred_layer_3/MatMulMatMulpred_layer_2/Relu:activations:0*pred_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#pred_layer_3/BiasAdd/ReadVariableOpReadVariableOp,pred_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
pred_layer_3/BiasAddBiasAddpred_layer_3/MatMul:product:0+pred_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
pred_layer_3/ReluRelupred_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+ctrl_layer_1_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype0
&ctrl_layer_1/kernel/Regularizer/SquareSquare=ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	(v
%ctrl_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_1/kernel/Regularizer/SumSum*ctrl_layer_1/kernel/Regularizer/Square:y:0.ctrl_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_1/kernel/Regularizer/mulMul.ctrl_layer_1/kernel/Regularizer/mul/x:output:0,ctrl_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp,ctrl_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_1/bias/Regularizer/SquareSquare;ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_1/bias/Regularizer/SumSum(ctrl_layer_1/bias/Regularizer/Square:y:0,ctrl_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_1/bias/Regularizer/mulMul,ctrl_layer_1/bias/Regularizer/mul/x:output:0*ctrl_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+ctrl_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&ctrl_layer_2/kernel/Regularizer/SquareSquare=ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%ctrl_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_2/kernel/Regularizer/SumSum*ctrl_layer_2/kernel/Regularizer/Square:y:0.ctrl_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_2/kernel/Regularizer/mulMul.ctrl_layer_2/kernel/Regularizer/mul/x:output:0,ctrl_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp,ctrl_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_2/bias/Regularizer/SquareSquare;ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_2/bias/Regularizer/SumSum(ctrl_layer_2/bias/Regularizer/Square:y:0,ctrl_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_2/bias/Regularizer/mulMul,ctrl_layer_2/bias/Regularizer/mul/x:output:0*ctrl_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+ctrl_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&ctrl_layer_3/kernel/Regularizer/SquareSquare=ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%ctrl_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_3/kernel/Regularizer/SumSum*ctrl_layer_3/kernel/Regularizer/Square:y:0.ctrl_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_3/kernel/Regularizer/mulMul.ctrl_layer_3/kernel/Regularizer/mul/x:output:0,ctrl_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp,ctrl_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$ctrl_layer_3/bias/Regularizer/SquareSquare;ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#ctrl_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_3/bias/Regularizer/SumSum(ctrl_layer_3/bias/Regularizer/Square:y:0,ctrl_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_3/bias/Regularizer/mulMul,ctrl_layer_3/bias/Regularizer/mul/x:output:0*ctrl_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+pred_layer_1_matmul_readvariableop_resource*
_output_shapes
:	)*
dtype0
&pred_layer_1/kernel/Regularizer/SquareSquare=pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	)v
%pred_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_1/kernel/Regularizer/SumSum*pred_layer_1/kernel/Regularizer/Square:y:0.pred_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_1/kernel/Regularizer/mulMul.pred_layer_1/kernel/Regularizer/mul/x:output:0,pred_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp,pred_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_1/bias/Regularizer/SquareSquare;pred_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_1/bias/Regularizer/SumSum(pred_layer_1/bias/Regularizer/Square:y:0,pred_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_1/bias/Regularizer/mulMul,pred_layer_1/bias/Regularizer/mul/x:output:0*pred_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: £
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+pred_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&pred_layer_2/kernel/Regularizer/SquareSquare=pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%pred_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_2/kernel/Regularizer/SumSum*pred_layer_2/kernel/Regularizer/Square:y:0.pred_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_2/kernel/Regularizer/mulMul.pred_layer_2/kernel/Regularizer/mul/x:output:0,pred_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp,pred_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_2/bias/Regularizer/SquareSquare;pred_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_2/bias/Regularizer/SumSum(pred_layer_2/bias/Regularizer/Square:y:0,pred_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_2/bias/Regularizer/mulMul,pred_layer_2/bias/Regularizer/mul/x:output:0*pred_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¢
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+pred_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&pred_layer_3/kernel/Regularizer/SquareSquare=pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%pred_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_3/kernel/Regularizer/SumSum*pred_layer_3/kernel/Regularizer/Square:y:0.pred_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_3/kernel/Regularizer/mulMul.pred_layer_3/kernel/Regularizer/mul/x:output:0,pred_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp,pred_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$pred_layer_3/bias/Regularizer/SquareSquare;pred_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#pred_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_3/bias/Regularizer/SumSum(pred_layer_3/bias/Regularizer/Square:y:0,pred_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_3/bias/Regularizer/mulMul,pred_layer_3/bias/Regularizer/mul/x:output:0*pred_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentitypred_layer_3/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp

Identity_1Identityctrl_layer_3/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà

NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp$^ctrl_layer_1/BiasAdd/ReadVariableOp#^ctrl_layer_1/MatMul/ReadVariableOp4^ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp$^ctrl_layer_2/BiasAdd/ReadVariableOp#^ctrl_layer_2/MatMul/ReadVariableOp4^ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp$^ctrl_layer_3/BiasAdd/ReadVariableOp#^ctrl_layer_3/MatMul/ReadVariableOp4^ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp$^pred_layer_1/BiasAdd/ReadVariableOp#^pred_layer_1/MatMul/ReadVariableOp4^pred_layer_1/bias/Regularizer/Square/ReadVariableOp6^pred_layer_1/kernel/Regularizer/Square/ReadVariableOp$^pred_layer_2/BiasAdd/ReadVariableOp#^pred_layer_2/MatMul/ReadVariableOp4^pred_layer_2/bias/Regularizer/Square/ReadVariableOp6^pred_layer_2/kernel/Regularizer/Square/ReadVariableOp$^pred_layer_3/BiasAdd/ReadVariableOp#^pred_layer_3/MatMul/ReadVariableOp4^pred_layer_3/bias/Regularizer/Square/ReadVariableOp6^pred_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2J
#ctrl_layer_1/BiasAdd/ReadVariableOp#ctrl_layer_1/BiasAdd/ReadVariableOp2H
"ctrl_layer_1/MatMul/ReadVariableOp"ctrl_layer_1/MatMul/ReadVariableOp2j
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp2J
#ctrl_layer_2/BiasAdd/ReadVariableOp#ctrl_layer_2/BiasAdd/ReadVariableOp2H
"ctrl_layer_2/MatMul/ReadVariableOp"ctrl_layer_2/MatMul/ReadVariableOp2j
3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp2J
#ctrl_layer_3/BiasAdd/ReadVariableOp#ctrl_layer_3/BiasAdd/ReadVariableOp2H
"ctrl_layer_3/MatMul/ReadVariableOp"ctrl_layer_3/MatMul/ReadVariableOp2j
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp2J
#pred_layer_1/BiasAdd/ReadVariableOp#pred_layer_1/BiasAdd/ReadVariableOp2H
"pred_layer_1/MatMul/ReadVariableOp"pred_layer_1/MatMul/ReadVariableOp2j
3pred_layer_1/bias/Regularizer/Square/ReadVariableOp3pred_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp2J
#pred_layer_2/BiasAdd/ReadVariableOp#pred_layer_2/BiasAdd/ReadVariableOp2H
"pred_layer_2/MatMul/ReadVariableOp"pred_layer_2/MatMul/ReadVariableOp2j
3pred_layer_2/bias/Regularizer/Square/ReadVariableOp3pred_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp2J
#pred_layer_3/BiasAdd/ReadVariableOp#pred_layer_3/BiasAdd/ReadVariableOp2H
"pred_layer_3/MatMul/ReadVariableOp"pred_layer_3/MatMul/ReadVariableOp2j
3pred_layer_3/bias/Regularizer/Square/ReadVariableOp3pred_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

è
H__inference_pred_layer_3_layer_call_and_return_conditional_losses_120426

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3pred_layer_3/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&pred_layer_3/kernel/Regularizer/SquareSquare=pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%pred_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_3/kernel/Regularizer/SumSum*pred_layer_3/kernel/Regularizer/Square:y:0.pred_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_3/kernel/Regularizer/mulMul.pred_layer_3/kernel/Regularizer/mul/x:output:0,pred_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$pred_layer_3/bias/Regularizer/SquareSquare;pred_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#pred_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_3/bias/Regularizer/SumSum(pred_layer_3/bias/Regularizer/Square:y:0,pred_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_3/bias/Regularizer/mulMul,pred_layer_3/bias/Regularizer/mul/x:output:0*pred_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^pred_layer_3/bias/Regularizer/Square/ReadVariableOp6^pred_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3pred_layer_3/bias/Regularizer/Square/ReadVariableOp3pred_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp5pred_layer_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

»
__inference_loss_fn_2_121947R
>ctrl_layer_2_kernel_regularizer_square_readvariableop_resource:

identity¢5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp¶
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>ctrl_layer_2_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype0
&ctrl_layer_2/kernel/Regularizer/SquareSquare=ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%ctrl_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_2/kernel/Regularizer/SumSum*ctrl_layer_2/kernel/Regularizer/Square:y:0.ctrl_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_2/kernel/Regularizer/mulMul.ctrl_layer_2/kernel/Regularizer/mul/x:output:0,ctrl_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentity'ctrl_layer_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_2/kernel/Regularizer/Square/ReadVariableOp
Ë
²
__inference_loss_fn_1_121936K
<ctrl_layer_1_bias_regularizer_square_readvariableop_resource:	
identity¢3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp­
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp<ctrl_layer_1_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype0
$ctrl_layer_1/bias/Regularizer/SquareSquare;ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#ctrl_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_1/bias/Regularizer/SumSum(ctrl_layer_1/bias/Regularizer/Square:y:0,ctrl_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_1/bias/Regularizer/mulMul,ctrl_layer_1/bias/Regularizer/mul/x:output:0*ctrl_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%ctrl_layer_1/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_1/bias/Regularizer/Square/ReadVariableOp

è
H__inference_ctrl_layer_3_layer_call_and_return_conditional_losses_120321

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&ctrl_layer_3/kernel/Regularizer/SquareSquare=ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%ctrl_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_3/kernel/Regularizer/SumSum*ctrl_layer_3/kernel/Regularizer/Square:y:0.ctrl_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_3/kernel/Regularizer/mulMul.ctrl_layer_3/kernel/Regularizer/mul/x:output:0,ctrl_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$ctrl_layer_3/bias/Regularizer/SquareSquare;ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#ctrl_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_3/bias/Regularizer/SumSum(ctrl_layer_3/bias/Regularizer/Square:y:0,ctrl_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_3/bias/Regularizer/mulMul,ctrl_layer_3/bias/Regularizer/mul/x:output:0*ctrl_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

é
H__inference_pred_layer_1_layer_call_and_return_conditional_losses_120368

inputs1
matmul_readvariableop_resource:	).
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3pred_layer_1/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	)*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	)*
dtype0
&pred_layer_1/kernel/Regularizer/SquareSquare=pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	)v
%pred_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_1/kernel/Regularizer/SumSum*pred_layer_1/kernel/Regularizer/Square:y:0.pred_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_1/kernel/Regularizer/mulMul.pred_layer_1/kernel/Regularizer/mul/x:output:0,pred_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_1/bias/Regularizer/SquareSquare;pred_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_1/bias/Regularizer/SumSum(pred_layer_1/bias/Regularizer/Square:y:0,pred_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_1/bias/Regularizer/mulMul,pred_layer_1/bias/Regularizer/mul/x:output:0*pred_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^pred_layer_1/bias/Regularizer/Square/ReadVariableOp6^pred_layer_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ): : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3pred_layer_1/bias/Regularizer/Square/ReadVariableOp3pred_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)
 
_user_specified_nameinputs
Í

-__inference_pred_layer_3_layer_call_fn_121891

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_pred_layer_3_layer_call_and_return_conditional_losses_120426o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
²
__inference_loss_fn_11_122046J
<pred_layer_3_bias_regularizer_square_readvariableop_resource:
identity¢3pred_layer_3/bias/Regularizer/Square/ReadVariableOp¬
3pred_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp<pred_layer_3_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype0
$pred_layer_3/bias/Regularizer/SquareSquare;pred_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#pred_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_3/bias/Regularizer/SumSum(pred_layer_3/bias/Regularizer/Square:y:0,pred_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_3/bias/Regularizer/mulMul,pred_layer_3/bias/Regularizer/mul/x:output:0*pred_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%pred_layer_3/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^pred_layer_3/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3pred_layer_3/bias/Regularizer/Square/ReadVariableOp3pred_layer_3/bias/Regularizer/Square/ReadVariableOp
Ë
²
__inference_loss_fn_9_122024K
<pred_layer_2_bias_regularizer_square_readvariableop_resource:	
identity¢3pred_layer_2/bias/Regularizer/Square/ReadVariableOp­
3pred_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp<pred_layer_2_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_2/bias/Regularizer/SquareSquare;pred_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_2/bias/Regularizer/SumSum(pred_layer_2/bias/Regularizer/Square:y:0,pred_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_2/bias/Regularizer/mulMul,pred_layer_2/bias/Regularizer/mul/x:output:0*pred_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
IdentityIdentity%pred_layer_2/bias/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^pred_layer_2/bias/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3pred_layer_2/bias/Regularizer/Square/ReadVariableOp3pred_layer_2/bias/Regularizer/Square/ReadVariableOp
Ñ

-__inference_ctrl_layer_2_layer_call_fn_121622

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_2_layer_call_and_return_conditional_losses_120292p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
è
O__inference_batch_normalization_layer_call_and_return_conditional_losses_120222

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í

-__inference_ctrl_layer_3_layer_call_fn_121666

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_3_layer_call_and_return_conditional_losses_120321o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
Ï
4__inference_batch_normalization_layer_call_fn_121715

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_120222o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Åf

"__inference__traced_restore_122236
file_prefix7
$assignvariableop_ctrl_layer_1_kernel:	(3
$assignvariableop_1_ctrl_layer_1_bias:	:
&assignvariableop_2_ctrl_layer_2_kernel:
3
$assignvariableop_3_ctrl_layer_2_bias:	9
&assignvariableop_4_ctrl_layer_3_kernel:	2
$assignvariableop_5_ctrl_layer_3_bias::
,assignvariableop_6_batch_normalization_gamma:9
+assignvariableop_7_batch_normalization_beta:@
2assignvariableop_8_batch_normalization_moving_mean:D
6assignvariableop_9_batch_normalization_moving_variance::
'assignvariableop_10_pred_layer_1_kernel:	)4
%assignvariableop_11_pred_layer_1_bias:	;
'assignvariableop_12_pred_layer_2_kernel:
4
%assignvariableop_13_pred_layer_2_bias:	:
'assignvariableop_14_pred_layer_3_kernel:	3
%assignvariableop_15_pred_layer_3_bias:#
assignvariableop_16_total: #
assignvariableop_17_count: %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: %
assignvariableop_20_total_2: %
assignvariableop_21_count_2: %
assignvariableop_22_total_3: %
assignvariableop_23_count_3: %
assignvariableop_24_total_4: %
assignvariableop_25_count_4: 
identity_27¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¨
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Î
valueÄBÁB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¦
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ¦
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp$assignvariableop_ctrl_layer_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_ctrl_layer_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp&assignvariableop_2_ctrl_layer_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_ctrl_layer_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp&assignvariableop_4_ctrl_layer_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_ctrl_layer_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp'assignvariableop_10_pred_layer_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp%assignvariableop_11_pred_layer_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp'assignvariableop_12_pred_layer_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_pred_layer_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp'assignvariableop_14_pred_layer_3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp%assignvariableop_15_pred_layer_3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_3Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_3Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_4Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_4Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: ø
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_25AssignVariableOp_252(
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
±
¥
$__inference_signature_wrapper_121557
input_1
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	)

unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_120151o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!
_user_specified_name	input_1
è\
¡
!__inference__wrapped_model_120151
input_1D
1model_ctrl_layer_1_matmul_readvariableop_resource:	(A
2model_ctrl_layer_1_biasadd_readvariableop_resource:	E
1model_ctrl_layer_2_matmul_readvariableop_resource:
A
2model_ctrl_layer_2_biasadd_readvariableop_resource:	D
1model_ctrl_layer_3_matmul_readvariableop_resource:	@
2model_ctrl_layer_3_biasadd_readvariableop_resource:I
;model_batch_normalization_batchnorm_readvariableop_resource:M
?model_batch_normalization_batchnorm_mul_readvariableop_resource:K
=model_batch_normalization_batchnorm_readvariableop_1_resource:K
=model_batch_normalization_batchnorm_readvariableop_2_resource:D
1model_pred_layer_1_matmul_readvariableop_resource:	)A
2model_pred_layer_1_biasadd_readvariableop_resource:	E
1model_pred_layer_2_matmul_readvariableop_resource:
A
2model_pred_layer_2_biasadd_readvariableop_resource:	D
1model_pred_layer_3_matmul_readvariableop_resource:	@
2model_pred_layer_3_biasadd_readvariableop_resource:
identity

identity_1¢2model/batch_normalization/batchnorm/ReadVariableOp¢4model/batch_normalization/batchnorm/ReadVariableOp_1¢4model/batch_normalization/batchnorm/ReadVariableOp_2¢6model/batch_normalization/batchnorm/mul/ReadVariableOp¢)model/ctrl_layer_1/BiasAdd/ReadVariableOp¢(model/ctrl_layer_1/MatMul/ReadVariableOp¢)model/ctrl_layer_2/BiasAdd/ReadVariableOp¢(model/ctrl_layer_2/MatMul/ReadVariableOp¢)model/ctrl_layer_3/BiasAdd/ReadVariableOp¢(model/ctrl_layer_3/MatMul/ReadVariableOp¢)model/pred_layer_1/BiasAdd/ReadVariableOp¢(model/pred_layer_1/MatMul/ReadVariableOp¢)model/pred_layer_2/BiasAdd/ReadVariableOp¢(model/pred_layer_2/MatMul/ReadVariableOp¢)model/pred_layer_3/BiasAdd/ReadVariableOp¢(model/pred_layer_3/MatMul/ReadVariableOp
(model/ctrl_layer_1/MatMul/ReadVariableOpReadVariableOp1model_ctrl_layer_1_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype0
model/ctrl_layer_1/MatMulMatMulinput_10model/ctrl_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/ctrl_layer_1/BiasAdd/ReadVariableOpReadVariableOp2model_ctrl_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
model/ctrl_layer_1/BiasAddBiasAdd#model/ctrl_layer_1/MatMul:product:01model/ctrl_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
model/ctrl_layer_1/ReluRelu#model/ctrl_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model/ctrl_layer_2/MatMul/ReadVariableOpReadVariableOp1model_ctrl_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¯
model/ctrl_layer_2/MatMulMatMul%model/ctrl_layer_1/Relu:activations:00model/ctrl_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/ctrl_layer_2/BiasAdd/ReadVariableOpReadVariableOp2model_ctrl_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
model/ctrl_layer_2/BiasAddBiasAdd#model/ctrl_layer_2/MatMul:product:01model/ctrl_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
model/ctrl_layer_2/ReluRelu#model/ctrl_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model/ctrl_layer_3/MatMul/ReadVariableOpReadVariableOp1model_ctrl_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
model/ctrl_layer_3/MatMulMatMul%model/ctrl_layer_2/Relu:activations:00model/ctrl_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/ctrl_layer_3/BiasAdd/ReadVariableOpReadVariableOp2model_ctrl_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¯
model/ctrl_layer_3/BiasAddBiasAdd#model/ctrl_layer_3/MatMul:product:01model/ctrl_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
model/ctrl_layer_3/ReluRelu#model/ctrl_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Å
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:²
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Â
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:¶
)model/batch_normalization/batchnorm/mul_1Mul%model/ctrl_layer_3/Relu:activations:0+model/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0À
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:®
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0À
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:À
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¿
model/concatenate/concatConcatV2input_1-model/batch_normalization/batchnorm/add_1:z:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)
(model/pred_layer_1/MatMul/ReadVariableOpReadVariableOp1model_pred_layer_1_matmul_readvariableop_resource*
_output_shapes
:	)*
dtype0«
model/pred_layer_1/MatMulMatMul!model/concatenate/concat:output:00model/pred_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/pred_layer_1/BiasAdd/ReadVariableOpReadVariableOp2model_pred_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
model/pred_layer_1/BiasAddBiasAdd#model/pred_layer_1/MatMul:product:01model/pred_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
model/pred_layer_1/ReluRelu#model/pred_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model/pred_layer_2/MatMul/ReadVariableOpReadVariableOp1model_pred_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¯
model/pred_layer_2/MatMulMatMul%model/pred_layer_1/Relu:activations:00model/pred_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/pred_layer_2/BiasAdd/ReadVariableOpReadVariableOp2model_pred_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
model/pred_layer_2/BiasAddBiasAdd#model/pred_layer_2/MatMul:product:01model/pred_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
model/pred_layer_2/ReluRelu#model/pred_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model/pred_layer_3/MatMul/ReadVariableOpReadVariableOp1model_pred_layer_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0®
model/pred_layer_3/MatMulMatMul%model/pred_layer_2/Relu:activations:00model/pred_layer_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/pred_layer_3/BiasAdd/ReadVariableOpReadVariableOp2model_pred_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¯
model/pred_layer_3/BiasAddBiasAdd#model/pred_layer_3/MatMul:product:01model/pred_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
model/pred_layer_3/ReluRelu#model/pred_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
IdentityIdentity%model/ctrl_layer_3/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv

Identity_1Identity%model/pred_layer_3/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp*^model/ctrl_layer_1/BiasAdd/ReadVariableOp)^model/ctrl_layer_1/MatMul/ReadVariableOp*^model/ctrl_layer_2/BiasAdd/ReadVariableOp)^model/ctrl_layer_2/MatMul/ReadVariableOp*^model/ctrl_layer_3/BiasAdd/ReadVariableOp)^model/ctrl_layer_3/MatMul/ReadVariableOp*^model/pred_layer_1/BiasAdd/ReadVariableOp)^model/pred_layer_1/MatMul/ReadVariableOp*^model/pred_layer_2/BiasAdd/ReadVariableOp)^model/pred_layer_2/MatMul/ReadVariableOp*^model/pred_layer_3/BiasAdd/ReadVariableOp)^model/pred_layer_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2V
)model/ctrl_layer_1/BiasAdd/ReadVariableOp)model/ctrl_layer_1/BiasAdd/ReadVariableOp2T
(model/ctrl_layer_1/MatMul/ReadVariableOp(model/ctrl_layer_1/MatMul/ReadVariableOp2V
)model/ctrl_layer_2/BiasAdd/ReadVariableOp)model/ctrl_layer_2/BiasAdd/ReadVariableOp2T
(model/ctrl_layer_2/MatMul/ReadVariableOp(model/ctrl_layer_2/MatMul/ReadVariableOp2V
)model/ctrl_layer_3/BiasAdd/ReadVariableOp)model/ctrl_layer_3/BiasAdd/ReadVariableOp2T
(model/ctrl_layer_3/MatMul/ReadVariableOp(model/ctrl_layer_3/MatMul/ReadVariableOp2V
)model/pred_layer_1/BiasAdd/ReadVariableOp)model/pred_layer_1/BiasAdd/ReadVariableOp2T
(model/pred_layer_1/MatMul/ReadVariableOp(model/pred_layer_1/MatMul/ReadVariableOp2V
)model/pred_layer_2/BiasAdd/ReadVariableOp)model/pred_layer_2/BiasAdd/ReadVariableOp2T
(model/pred_layer_2/MatMul/ReadVariableOp(model/pred_layer_2/MatMul/ReadVariableOp2V
)model/pred_layer_3/BiasAdd/ReadVariableOp)model/pred_layer_3/BiasAdd/ReadVariableOp2T
(model/pred_layer_3/MatMul/ReadVariableOp(model/pred_layer_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!
_user_specified_name	input_1
Ó
§
&__inference_model_layer_call_fn_120543
input_1
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	)

unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_120506o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!
_user_specified_name	input_1
¨
X
,__inference_concatenate_layer_call_fn_121775
inputs_0
inputs_1
identity¿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_120343`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

é
H__inference_pred_layer_1_layer_call_and_return_conditional_losses_121826

inputs1
matmul_readvariableop_resource:	).
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3pred_layer_1/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	)*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	)*
dtype0
&pred_layer_1/kernel/Regularizer/SquareSquare=pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	)v
%pred_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_1/kernel/Regularizer/SumSum*pred_layer_1/kernel/Regularizer/Square:y:0.pred_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_1/kernel/Regularizer/mulMul.pred_layer_1/kernel/Regularizer/mul/x:output:0,pred_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_1/bias/Regularizer/SquareSquare;pred_layer_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_1/bias/Regularizer/SumSum(pred_layer_1/bias/Regularizer/Square:y:0,pred_layer_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_1/bias/Regularizer/mulMul,pred_layer_1/bias/Regularizer/mul/x:output:0*pred_layer_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^pred_layer_1/bias/Regularizer/Square/ReadVariableOp6^pred_layer_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ): : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3pred_layer_1/bias/Regularizer/Square/ReadVariableOp3pred_layer_1/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp5pred_layer_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ)
 
_user_specified_nameinputs
Î

-__inference_ctrl_layer_1_layer_call_fn_121578

inputs
unknown:	(
	unknown_0:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_ctrl_layer_1_layer_call_and_return_conditional_losses_120263p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs

è
H__inference_ctrl_layer_3_layer_call_and_return_conditional_losses_121689

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp¢5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&ctrl_layer_3/kernel/Regularizer/SquareSquare=ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%ctrl_layer_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_3/kernel/Regularizer/SumSum*ctrl_layer_3/kernel/Regularizer/Square:y:0.ctrl_layer_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_3/kernel/Regularizer/mulMul.ctrl_layer_3/kernel/Regularizer/mul/x:output:0,ctrl_layer_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
$ctrl_layer_3/bias/Regularizer/SquareSquare;ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:m
#ctrl_layer_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!ctrl_layer_3/bias/Regularizer/SumSum(ctrl_layer_3/bias/Regularizer/Square:y:0,ctrl_layer_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#ctrl_layer_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!ctrl_layer_3/bias/Regularizer/mulMul,ctrl_layer_3/bias/Regularizer/mul/x:output:0*ctrl_layer_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp6^ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp3ctrl_layer_3/bias/Regularizer/Square/ReadVariableOp2n
5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ê
H__inference_pred_layer_2_layer_call_and_return_conditional_losses_121870

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢3pred_layer_2/bias/Regularizer/Square/ReadVariableOp¢5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
&pred_layer_2/kernel/Regularizer/SquareSquare=pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
v
%pred_layer_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#pred_layer_2/kernel/Regularizer/SumSum*pred_layer_2/kernel/Regularizer/Square:y:0.pred_layer_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%pred_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#pred_layer_2/kernel/Regularizer/mulMul.pred_layer_2/kernel/Regularizer/mul/x:output:0,pred_layer_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
3pred_layer_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
$pred_layer_2/bias/Regularizer/SquareSquare;pred_layer_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:m
#pred_layer_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
!pred_layer_2/bias/Regularizer/SumSum(pred_layer_2/bias/Regularizer/Square:y:0,pred_layer_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: h
#pred_layer_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:£
!pred_layer_2/bias/Regularizer/mulMul,pred_layer_2/bias/Regularizer/mul/x:output:0*pred_layer_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^pred_layer_2/bias/Regularizer/Square/ReadVariableOp6^pred_layer_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3pred_layer_2/bias/Regularizer/Square/ReadVariableOp3pred_layer_2/bias/Regularizer/Square/ReadVariableOp2n
5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp5pred_layer_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

º
__inference_loss_fn_0_121925Q
>ctrl_layer_1_kernel_regularizer_square_readvariableop_resource:	(
identity¢5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpµ
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>ctrl_layer_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	(*
dtype0
&ctrl_layer_1/kernel/Regularizer/SquareSquare=ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	(v
%ctrl_layer_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#ctrl_layer_1/kernel/Regularizer/SumSum*ctrl_layer_1/kernel/Regularizer/Square:y:0.ctrl_layer_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%ctrl_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:©
#ctrl_layer_1/kernel/Regularizer/mulMul.ctrl_layer_1/kernel/Regularizer/mul/x:output:0,ctrl_layer_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentity'ctrl_layer_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp5ctrl_layer_1/kernel/Regularizer/Square/ReadVariableOp
Ñ
§
&__inference_model_layer_call_fn_120844
input_1
unknown:	(
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:	)

unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:
identity

identity_1¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_120768o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ(: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!
_user_specified_name	input_1"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ñ
serving_defaultÝ
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ(@
ctrl_layer_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ@
pred_layer_30
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ä´

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

loss
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
»

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
+axis
	,gamma
-beta
.moving_mean
/moving_variance
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
»

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper

0
1
2
3
#4
$5
,6
-7
.8
/9
<10
=11
D12
E13
L14
M15"
trackable_list_wrapper

0
1
2
3
#4
$5
,6
-7
<8
=9
D10
E11
L12
M13"
trackable_list_wrapper
v
T0
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11"
trackable_list_wrapper
Ê
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
æ2ã
&__inference_model_layer_call_fn_120543
&__inference_model_layer_call_fn_121189
&__inference_model_layer_call_fn_121228
&__inference_model_layer_call_fn_120844À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_model_layer_call_and_return_conditional_losses_121365
A__inference_model_layer_call_and_return_conditional_losses_121516
A__inference_model_layer_call_and_return_conditional_losses_120961
A__inference_model_layer_call_and_return_conditional_losses_121078À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÌBÉ
!__inference__wrapped_model_120151input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
eserving_default"
signature_map
&:$	(2ctrl_layer_1/kernel
 :2ctrl_layer_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_ctrl_layer_1_layer_call_fn_121578¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_ctrl_layer_1_layer_call_and_return_conditional_losses_121601¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%
2ctrl_layer_2/kernel
 :2ctrl_layer_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_ctrl_layer_2_layer_call_fn_121622¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_ctrl_layer_2_layer_call_and_return_conditional_losses_121645¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$	2ctrl_layer_3/kernel
:2ctrl_layer_3/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_ctrl_layer_3_layer_call_fn_121666¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_ctrl_layer_3_layer_call_and_return_conditional_losses_121689¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
,0
-1
.2
/3"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
¦2£
4__inference_batch_normalization_layer_call_fn_121702
4__inference_batch_normalization_layer_call_fn_121715´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
O__inference_batch_normalization_layer_call_and_return_conditional_losses_121735
O__inference_batch_normalization_layer_call_and_return_conditional_losses_121769´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_concatenate_layer_call_fn_121775¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_concatenate_layer_call_and_return_conditional_losses_121782¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$	)2pred_layer_1/kernel
 :2pred_layer_1/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
±
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_pred_layer_1_layer_call_fn_121803¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_pred_layer_1_layer_call_and_return_conditional_losses_121826¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
':%
2pred_layer_2/kernel
 :2pred_layer_2/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_pred_layer_2_layer_call_fn_121847¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_pred_layer_2_layer_call_and_return_conditional_losses_121870¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
&:$	2pred_layer_3/kernel
:2pred_layer_3/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_pred_layer_3_layer_call_fn_121891¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_pred_layer_3_layer_call_and_return_conditional_losses_121914¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³2°
__inference_loss_fn_0_121925
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_1_121936
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_2_121947
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_3_121958
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_4_121969
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_5_121980
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_6_121991
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_7_122002
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_8_122013
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_9_122024
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_10_122035
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_11_122046
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
.
.0
/1"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
H
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ËBÈ
$__inference_signature_wrapper_121557input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
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
V0
W1"
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
X0
Y1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
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
Z0
[1"
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
\0
]1"
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
^0
_1"
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
R

total

count
	variables
	keras_api"
_tf_keras_metric
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

 count
¡
_fn_kwargs
¢	variables
£	keras_api"
_tf_keras_metric
c

¤total

¥count
¦
_fn_kwargs
§	variables
¨	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
 1"
trackable_list_wrapper
.
¢	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¤0
¥1"
trackable_list_wrapper
.
§	variables"
_generic_user_objectß
!__inference__wrapped_model_120151¹#$/,.-<=DELM0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ(
ª "sªp
6
ctrl_layer_3&#
ctrl_layer_3ÿÿÿÿÿÿÿÿÿ
6
pred_layer_3&#
pred_layer_3ÿÿÿÿÿÿÿÿÿµ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_121735b/,.-3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_121769b./,-3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
4__inference_batch_normalization_layer_call_fn_121702U/,.-3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
4__inference_batch_normalization_layer_call_fn_121715U./,-3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÏ
G__inference_concatenate_layer_call_and_return_conditional_losses_121782Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ(
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ)
 ¦
,__inference_concatenate_layer_call_fn_121775vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ(
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ)©
H__inference_ctrl_layer_1_layer_call_and_return_conditional_losses_121601]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_ctrl_layer_1_layer_call_fn_121578P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª "ÿÿÿÿÿÿÿÿÿª
H__inference_ctrl_layer_2_layer_call_and_return_conditional_losses_121645^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_ctrl_layer_2_layer_call_fn_121622Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
H__inference_ctrl_layer_3_layer_call_and_return_conditional_losses_121689]#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_ctrl_layer_3_layer_call_fn_121666P#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ;
__inference_loss_fn_0_121925¢

¢ 
ª " <
__inference_loss_fn_10_122035L¢

¢ 
ª " <
__inference_loss_fn_11_122046M¢

¢ 
ª " ;
__inference_loss_fn_1_121936¢

¢ 
ª " ;
__inference_loss_fn_2_121947¢

¢ 
ª " ;
__inference_loss_fn_3_121958¢

¢ 
ª " ;
__inference_loss_fn_4_121969#¢

¢ 
ª " ;
__inference_loss_fn_5_121980$¢

¢ 
ª " ;
__inference_loss_fn_6_121991<¢

¢ 
ª " ;
__inference_loss_fn_7_122002=¢

¢ 
ª " ;
__inference_loss_fn_8_122013D¢

¢ 
ª " ;
__inference_loss_fn_9_122024E¢

¢ 
ª " ß
A__inference_model_layer_call_and_return_conditional_losses_120961#$/,.-<=DELM8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ(
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ß
A__inference_model_layer_call_and_return_conditional_losses_121078#$./,-<=DELM8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ(
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 Þ
A__inference_model_layer_call_and_return_conditional_losses_121365#$/,.-<=DELM7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ(
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 Þ
A__inference_model_layer_call_and_return_conditional_losses_121516#$./,-<=DELM7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ(
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ¶
&__inference_model_layer_call_fn_120543#$/,.-<=DELM8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ(
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ¶
&__inference_model_layer_call_fn_120844#$./,-<=DELM8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ(
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿµ
&__inference_model_layer_call_fn_121189#$/,.-<=DELM7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ(
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿµ
&__inference_model_layer_call_fn_121228#$./,-<=DELM7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ(
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ©
H__inference_pred_layer_1_layer_call_and_return_conditional_losses_121826]<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ)
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_pred_layer_1_layer_call_fn_121803P<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ)
ª "ÿÿÿÿÿÿÿÿÿª
H__inference_pred_layer_2_layer_call_and_return_conditional_losses_121870^DE0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_pred_layer_2_layer_call_fn_121847QDE0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
H__inference_pred_layer_3_layer_call_and_return_conditional_losses_121914]LM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_pred_layer_3_layer_call_fn_121891PLM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿí
$__inference_signature_wrapper_121557Ä#$/,.-<=DELM;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ("sªp
6
ctrl_layer_3&#
ctrl_layer_3ÿÿÿÿÿÿÿÿÿ
6
pred_layer_3&#
pred_layer_3ÿÿÿÿÿÿÿÿÿ