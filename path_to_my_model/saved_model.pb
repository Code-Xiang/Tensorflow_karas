??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108??
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
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
r
lstm/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_namelstm/kernel
k
lstm/kernel/Read/ReadVariableOpReadVariableOplstm/kernel*
_output_shapes

:(*
dtype0
?
lstm/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*&
shared_namelstm/recurrent_kernel

)lstm/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/recurrent_kernel*
_output_shapes

:
(*
dtype0
j
	lstm/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_name	lstm/bias
c
lstm/bias/Read/ReadVariableOpReadVariableOp	lstm/bias*
_output_shapes
:(*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/lstm/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_nameAdam/lstm/kernel/m
y
&Adam/lstm/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/kernel/m*
_output_shapes

:(*
dtype0
?
Adam/lstm/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*-
shared_nameAdam/lstm/recurrent_kernel/m
?
0Adam/lstm/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/recurrent_kernel/m*
_output_shapes

:
(*
dtype0
x
Adam/lstm/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_nameAdam/lstm/bias/m
q
$Adam/lstm/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/bias/m*
_output_shapes
:(*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:
*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/lstm/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*#
shared_nameAdam/lstm/kernel/v
y
&Adam/lstm/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/kernel/v*
_output_shapes

:(*
dtype0
?
Adam/lstm/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
(*-
shared_nameAdam/lstm/recurrent_kernel/v
?
0Adam/lstm/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/recurrent_kernel/v*
_output_shapes

:
(*
dtype0
x
Adam/lstm/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*!
shared_nameAdam/lstm/bias/v
q
$Adam/lstm/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/bias/v*
_output_shapes
:(*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
 
l

cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratem2m3m4m5m6v7v8v9v:v;
 
#
0
1
2
3
4
#
0
1
2
3
4
?
layer_regularization_losses
non_trainable_variables

 layers
regularization_losses
	variables
trainable_variables
!metrics
 
~

kernel
recurrent_kernel
bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
 
 

0
1
2

0
1
2
?
&layer_regularization_losses
'non_trainable_variables

(layers
regularization_losses
	variables
trainable_variables
)metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
*layer_regularization_losses
+non_trainable_variables

,layers
regularization_losses
	variables
trainable_variables
-metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUElstm/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE	lstm/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
 

0
1
2

0
1
2
?
.layer_regularization_losses
/non_trainable_variables

0layers
"regularization_losses
#	variables
$trainable_variables
1metrics
 
 


0
 
 
 
 
 
 
 
 
 
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/lstm/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/lstm/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/lstm/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEAdam/lstm/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_inputPlaceholder*+
_output_shapes
:??????????*
dtype0* 
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputlstm/kernel	lstm/biaslstm/recurrent_kerneldense/kernel
dense/bias*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_signature_wrapper_9065
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOplstm/kernel/Read/ReadVariableOp)lstm/recurrent_kernel/Read/ReadVariableOplstm/bias/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp&Adam/lstm/kernel/m/Read/ReadVariableOp0Adam/lstm/recurrent_kernel/m/Read/ReadVariableOp$Adam/lstm/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp&Adam/lstm/kernel/v/Read/ReadVariableOp0Adam/lstm/recurrent_kernel/v/Read/ReadVariableOp$Adam/lstm/bias/v/Read/ReadVariableOpConst*!
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__traced_save_11072
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/kernellstm/recurrent_kernel	lstm/biasAdam/dense/kernel/mAdam/dense/bias/mAdam/lstm/kernel/mAdam/lstm/recurrent_kernel/mAdam/lstm/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/lstm/kernel/vAdam/lstm/recurrent_kernel/vAdam/lstm/bias/v* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_restore_11144??
?
?
while_body_8324
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5??StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:?????????
:?????????
:?????????
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_79282
StatefulPartitionedCall?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1f
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identityy

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1h

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::22
StatefulPartitionedCallStatefulPartitionedCall
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_9036

inputs'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_89372
lstm/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89692
dense/StatefulPartitionedCall?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
+
__inference_loss_fn_1_10988
identity?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Consti
IdentityIdentity&lstm/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
)__inference_sequential_layer_call_fn_9627

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_90132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?j
?
while_body_8794
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
split_readvariableop_resource_0%
!split_1_readvariableop_resource_0
readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource_0*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
split?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMul?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2?
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOp!split_1_readvariableop_resource_0*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicew
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2y
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3y
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource_0^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4y
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_8/yW
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: 2
add_8T
add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_9/y^
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: 2
add_9?
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"4
readvariableop_resourcereadvariableop_resource_0"D
split_1_readvariableop_resource!split_1_readvariableop_resource_0"@
split_readvariableop_resourcesplit_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp
?D
?
>__inference_lstm_layer_call_and_return_conditional_losses_8390

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:?????????
:?????????
:?????????
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_79282
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *
bodyR
while_body_8324*
condR
while_cond_8323*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs
??
?
?__inference_lstm_layer_call_and_return_conditional_losses_10730

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitv
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMulz
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1z
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2z
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3z
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4z
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_5z
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_6z
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcereadvariableop_resource^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *
bodyR
while_body_10587*
condR
while_cond_10586*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:??????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_7y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:??????????
2
transpose_1?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
?
?
while_cond_10317
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_10317___redundant_placeholder0-
)while_cond_10317___redundant_placeholder1-
)while_cond_10317___redundant_placeholder2-
)while_cond_10317___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
?0
?
__inference__traced_save_11072
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop*
&savev2_lstm_kernel_read_readvariableop4
0savev2_lstm_recurrent_kernel_read_readvariableop(
$savev2_lstm_bias_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop1
-savev2_adam_lstm_kernel_m_read_readvariableop;
7savev2_adam_lstm_recurrent_kernel_m_read_readvariableop/
+savev2_adam_lstm_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop1
-savev2_adam_lstm_kernel_v_read_readvariableop;
7savev2_adam_lstm_recurrent_kernel_v_read_readvariableop/
+savev2_adam_lstm_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d7dcc262851d4d5f913aa34b7615179a/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop&savev2_lstm_kernel_read_readvariableop0savev2_lstm_recurrent_kernel_read_readvariableop$savev2_lstm_bias_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop-savev2_adam_lstm_kernel_m_read_readvariableop7savev2_adam_lstm_recurrent_kernel_m_read_readvariableop+savev2_adam_lstm_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop-savev2_adam_lstm_kernel_v_read_readvariableop7savev2_adam_lstm_recurrent_kernel_v_read_readvariableop+savev2_adam_lstm_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
:: : : : : :(:
(:(:
::(:
(:(:
::(:
(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
?
$__inference_lstm_layer_call_fn_10192
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_83902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_9013

inputs'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_86682
lstm/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89692
dense/StatefulPartitionedCall?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_8969

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_10758

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?j
?
while_body_10318
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
split_readvariableop_resource_0%
!split_1_readvariableop_resource_0
readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource_0*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
split?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMul?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2?
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOp!split_1_readvariableop_resource_0*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicew
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2y
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3y
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource_0^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4y
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_8/yW
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: 2
add_8T
add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_9/y^
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: 2
add_9?
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"4
readvariableop_resourcereadvariableop_resource_0"D
split_1_readvariableop_resource!split_1_readvariableop_resource_0"@
split_readvariableop_resourcesplit_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp
?
?
$__inference_lstm_layer_call_fn_10738

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_86682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_8984

lstm_input'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_86682
lstm/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89692
dense/StatefulPartitionedCall?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
?D
?
>__inference_lstm_layer_call_and_return_conditional_losses_8267

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:?????????
:?????????
:?????????
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_78362
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *
bodyR
while_body_8201*
condR
while_cond_8200*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs
?U
?

!__inference__traced_restore_11144
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate"
assignvariableop_7_lstm_kernel,
(assignvariableop_8_lstm_recurrent_kernel 
assignvariableop_9_lstm_bias+
'assignvariableop_10_adam_dense_kernel_m)
%assignvariableop_11_adam_dense_bias_m*
&assignvariableop_12_adam_lstm_kernel_m4
0assignvariableop_13_adam_lstm_recurrent_kernel_m(
$assignvariableop_14_adam_lstm_bias_m+
'assignvariableop_15_adam_dense_kernel_v)
%assignvariableop_16_adam_dense_bias_v*
&assignvariableop_17_adam_lstm_kernel_v4
0assignvariableop_18_adam_lstm_recurrent_kernel_v(
$assignvariableop_19_adam_lstm_bias_v
identity_21??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_lstm_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_lstm_recurrent_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_lstm_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_dense_kernel_mIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_adam_dense_bias_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_lstm_kernel_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp0assignvariableop_13_adam_lstm_recurrent_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_adam_lstm_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp&assignvariableop_17_adam_lstm_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_lstm_recurrent_kernel_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_lstm_bias_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20?
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
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
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
?
while_cond_9763
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1,
(while_cond_9763___redundant_placeholder0,
(while_cond_9763___redundant_placeholder1,
(while_cond_9763___redundant_placeholder2,
(while_cond_9763___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
?
?
)__inference_lstm_cell_layer_call_fn_10969

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:?????????
:?????????
:?????????
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_78362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????
:?????????
:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
?
?
$__inference_lstm_layer_call_fn_10746

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_89372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?j
?
while_body_10033
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
split_readvariableop_resource_0%
!split_1_readvariableop_resource_0
readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource_0*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
split?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMul?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2?
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOp!split_1_readvariableop_resource_0*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicew
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2y
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3y
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource_0^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4y
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_8/yW
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: 2
add_8T
add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_9/y^
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: 2
add_9?
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"4
readvariableop_resourcereadvariableop_resource_0"D
split_1_readvariableop_resource!split_1_readvariableop_resource_0"@
split_readvariableop_resourcesplit_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp
?j
?
lstm_while_body_9467
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0#
split_readvariableop_resource_0%
!split_1_readvariableop_resource_0
readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource_0*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
split?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMul?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2?
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOp!split_1_readvariableop_resource_0*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicew
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_1y
MatMul_5MatMulplaceholder_2strided_slice_1:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2y
MatMul_6MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource_0^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3y
MatMul_7MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_8/yW
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: 2
add_8T
add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_9/yc
add_9AddV2lstm_while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: 2
add_9?
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitylstm_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0".
lstm_strided_slice_1lstm_strided_slice_1_0"4
readvariableop_resourcereadvariableop_resource_0"D
split_1_readvariableop_resource!split_1_readvariableop_resource_0"@
split_readvariableop_resourcesplit_readvariableop_resource_0"?
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp
?V
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_10955

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????
2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicer
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_1t
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1f
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2t
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3t
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????
:?????????
:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
?
?
"__inference_signature_wrapper_9065

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__wrapped_model_77042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
?
?
)__inference_sequential_layer_call_fn_9021

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_90132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
?
?
%__inference_dense_layer_call_fn_10765

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
??
?
>__inference_lstm_layer_call_and_return_conditional_losses_8668

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitv
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMulz
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1z
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2z
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3z
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4z
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_5z
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_6z
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcereadvariableop_resource^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *
bodyR
while_body_8525*
condR
while_cond_8524*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:??????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_7y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:??????????
2
transpose_1?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
??
?
?__inference_lstm_layer_call_and_return_conditional_losses_10461

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitv
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMulz
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1z
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2z
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3z
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4z
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_5z
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_6z
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcereadvariableop_resource^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *
bodyR
while_body_10318*
condR
while_cond_10317*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:??????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_7y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:??????????
2
transpose_1?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_9044

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_90362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_8997

lstm_input'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?lstm/StatefulPartitionedCall?
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_89372
lstm/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_89692
dense/StatefulPartitionedCall?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
?
?
while_cond_8793
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1,
(while_cond_8793___redundant_placeholder0,
(while_cond_8793___redundant_placeholder1,
(while_cond_8793___redundant_placeholder2,
(while_cond_8793___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
??
?
>__inference_lstm_layer_call_and_return_conditional_losses_8937

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitv
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMulz
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1z
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2z
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3z
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4z
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_5z
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_6z
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcereadvariableop_resource^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *
bodyR
while_body_8794*
condR
while_cond_8793*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:??????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_7y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:??????????
2
transpose_1?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
?
?
while_body_8201
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5??StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:?????????
:?????????
:?????????
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_78362
StatefulPartitionedCall?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1f
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identityy

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1h

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::22
StatefulPartitionedCallStatefulPartitionedCall
??
?
__inference__wrapped_model_7704

lstm_input1
-sequential_lstm_split_readvariableop_resource3
/sequential_lstm_split_1_readvariableop_resource+
'sequential_lstm_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?sequential/lstm/ReadVariableOp? sequential/lstm/ReadVariableOp_1? sequential/lstm/ReadVariableOp_2? sequential/lstm/ReadVariableOp_3?$sequential/lstm/split/ReadVariableOp?&sequential/lstm/split_1/ReadVariableOp?sequential/lstm/whileh
sequential/lstm/ShapeShape
lstm_input*
T0*
_output_shapes
:2
sequential/lstm/Shape?
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/lstm/strided_slice/stack?
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_1?
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_2?
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/lstm/strided_slice|
sequential/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
sequential/lstm/zeros/mul/y?
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/mul
sequential/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
sequential/lstm/zeros/Less/y?
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/Less?
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2 
sequential/lstm/zeros/packed/1?
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/lstm/zeros/packed
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros/Const?
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/zeros?
sequential/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
sequential/lstm/zeros_1/mul/y?
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/mul?
sequential/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2 
sequential/lstm/zeros_1/Less/y?
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/Less?
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2"
 sequential/lstm/zeros_1/packed/1?
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
sequential/lstm/zeros_1/packed?
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros_1/Const?
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/zeros_1?
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
sequential/lstm/transpose/perm?
sequential/lstm/transpose	Transpose
lstm_input'sequential/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:??????????2
sequential/lstm/transpose
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:2
sequential/lstm/Shape_1?
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_1/stack?
'sequential/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_1?
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_2?
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
sequential/lstm/strided_slice_1?
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+sequential/lstm/TensorArrayV2/element_shape?
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
sequential/lstm/TensorArrayV2?
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7sequential/lstm/TensorArrayUnstack/TensorListFromTensor?
%sequential/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_2/stack?
'sequential/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_1?
'sequential/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_2?
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2!
sequential/lstm/strided_slice_2p
sequential/lstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/Const?
sequential/lstm/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential/lstm/split/split_dim?
$sequential/lstm/split/ReadVariableOpReadVariableOp-sequential_lstm_split_readvariableop_resource*
_output_shapes

:(*
dtype02&
$sequential/lstm/split/ReadVariableOp?
sequential/lstm/splitSplit(sequential/lstm/split/split_dim:output:0,sequential/lstm/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
sequential/lstm/split?
sequential/lstm/MatMulMatMul(sequential/lstm/strided_slice_2:output:0sequential/lstm/split:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/MatMul?
sequential/lstm/MatMul_1MatMul(sequential/lstm/strided_slice_2:output:0sequential/lstm/split:output:1*
T0*'
_output_shapes
:?????????
2
sequential/lstm/MatMul_1?
sequential/lstm/MatMul_2MatMul(sequential/lstm/strided_slice_2:output:0sequential/lstm/split:output:2*
T0*'
_output_shapes
:?????????
2
sequential/lstm/MatMul_2?
sequential/lstm/MatMul_3MatMul(sequential/lstm/strided_slice_2:output:0sequential/lstm/split:output:3*
T0*'
_output_shapes
:?????????
2
sequential/lstm/MatMul_3t
sequential/lstm/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/Const_1?
!sequential/lstm/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/lstm/split_1/split_dim?
&sequential/lstm/split_1/ReadVariableOpReadVariableOp/sequential_lstm_split_1_readvariableop_resource*
_output_shapes
:(*
dtype02(
&sequential/lstm/split_1/ReadVariableOp?
sequential/lstm/split_1Split*sequential/lstm/split_1/split_dim:output:0.sequential/lstm/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2
sequential/lstm/split_1?
sequential/lstm/BiasAddBiasAdd sequential/lstm/MatMul:product:0 sequential/lstm/split_1:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/BiasAdd?
sequential/lstm/BiasAdd_1BiasAdd"sequential/lstm/MatMul_1:product:0 sequential/lstm/split_1:output:1*
T0*'
_output_shapes
:?????????
2
sequential/lstm/BiasAdd_1?
sequential/lstm/BiasAdd_2BiasAdd"sequential/lstm/MatMul_2:product:0 sequential/lstm/split_1:output:2*
T0*'
_output_shapes
:?????????
2
sequential/lstm/BiasAdd_2?
sequential/lstm/BiasAdd_3BiasAdd"sequential/lstm/MatMul_3:product:0 sequential/lstm/split_1:output:3*
T0*'
_output_shapes
:?????????
2
sequential/lstm/BiasAdd_3?
sequential/lstm/ReadVariableOpReadVariableOp'sequential_lstm_readvariableop_resource*
_output_shapes

:
(*
dtype02 
sequential/lstm/ReadVariableOp?
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%sequential/lstm/strided_slice_3/stack?
'sequential/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2)
'sequential/lstm/strided_slice_3/stack_1?
'sequential/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'sequential/lstm/strided_slice_3/stack_2?
sequential/lstm/strided_slice_3StridedSlice&sequential/lstm/ReadVariableOp:value:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2!
sequential/lstm/strided_slice_3?
sequential/lstm/MatMul_4MatMulsequential/lstm/zeros:output:0(sequential/lstm/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/MatMul_4?
sequential/lstm/addAddV2 sequential/lstm/BiasAdd:output:0"sequential/lstm/MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/addw
sequential/lstm/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
sequential/lstm/Const_2w
sequential/lstm/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/lstm/Const_3?
sequential/lstm/MulMulsequential/lstm/add:z:0 sequential/lstm/Const_2:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/Mul?
sequential/lstm/Add_1Addsequential/lstm/Mul:z:0 sequential/lstm/Const_3:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/Add_1?
'sequential/lstm/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'sequential/lstm/clip_by_value/Minimum/y?
%sequential/lstm/clip_by_value/MinimumMinimumsequential/lstm/Add_1:z:00sequential/lstm/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2'
%sequential/lstm/clip_by_value/Minimum?
sequential/lstm/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/lstm/clip_by_value/y?
sequential/lstm/clip_by_valueMaximum)sequential/lstm/clip_by_value/Minimum:z:0(sequential/lstm/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/clip_by_value?
 sequential/lstm/ReadVariableOp_1ReadVariableOp'sequential_lstm_readvariableop_resource^sequential/lstm/ReadVariableOp*
_output_shapes

:
(*
dtype02"
 sequential/lstm/ReadVariableOp_1?
%sequential/lstm/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2'
%sequential/lstm/strided_slice_4/stack?
'sequential/lstm/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'sequential/lstm/strided_slice_4/stack_1?
'sequential/lstm/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'sequential/lstm/strided_slice_4/stack_2?
sequential/lstm/strided_slice_4StridedSlice(sequential/lstm/ReadVariableOp_1:value:0.sequential/lstm/strided_slice_4/stack:output:00sequential/lstm/strided_slice_4/stack_1:output:00sequential/lstm/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2!
sequential/lstm/strided_slice_4?
sequential/lstm/MatMul_5MatMulsequential/lstm/zeros:output:0(sequential/lstm/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/MatMul_5?
sequential/lstm/add_2AddV2"sequential/lstm/BiasAdd_1:output:0"sequential/lstm/MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/add_2w
sequential/lstm/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
sequential/lstm/Const_4w
sequential/lstm/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/lstm/Const_5?
sequential/lstm/Mul_1Mulsequential/lstm/add_2:z:0 sequential/lstm/Const_4:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/Mul_1?
sequential/lstm/Add_3Addsequential/lstm/Mul_1:z:0 sequential/lstm/Const_5:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/Add_3?
)sequential/lstm/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)sequential/lstm/clip_by_value_1/Minimum/y?
'sequential/lstm/clip_by_value_1/MinimumMinimumsequential/lstm/Add_3:z:02sequential/lstm/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2)
'sequential/lstm/clip_by_value_1/Minimum?
!sequential/lstm/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/lstm/clip_by_value_1/y?
sequential/lstm/clip_by_value_1Maximum+sequential/lstm/clip_by_value_1/Minimum:z:0*sequential/lstm/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2!
sequential/lstm/clip_by_value_1?
sequential/lstm/mul_2Mul#sequential/lstm/clip_by_value_1:z:0 sequential/lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/mul_2?
 sequential/lstm/ReadVariableOp_2ReadVariableOp'sequential_lstm_readvariableop_resource!^sequential/lstm/ReadVariableOp_1*
_output_shapes

:
(*
dtype02"
 sequential/lstm/ReadVariableOp_2?
%sequential/lstm/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential/lstm/strided_slice_5/stack?
'sequential/lstm/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'sequential/lstm/strided_slice_5/stack_1?
'sequential/lstm/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'sequential/lstm/strided_slice_5/stack_2?
sequential/lstm/strided_slice_5StridedSlice(sequential/lstm/ReadVariableOp_2:value:0.sequential/lstm/strided_slice_5/stack:output:00sequential/lstm/strided_slice_5/stack_1:output:00sequential/lstm/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2!
sequential/lstm/strided_slice_5?
sequential/lstm/MatMul_6MatMulsequential/lstm/zeros:output:0(sequential/lstm/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/MatMul_6?
sequential/lstm/add_4AddV2"sequential/lstm/BiasAdd_2:output:0"sequential/lstm/MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/add_4?
sequential/lstm/TanhTanhsequential/lstm/add_4:z:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/Tanh?
sequential/lstm/mul_3Mul!sequential/lstm/clip_by_value:z:0sequential/lstm/Tanh:y:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/mul_3?
sequential/lstm/add_5AddV2sequential/lstm/mul_2:z:0sequential/lstm/mul_3:z:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/add_5?
 sequential/lstm/ReadVariableOp_3ReadVariableOp'sequential_lstm_readvariableop_resource!^sequential/lstm/ReadVariableOp_2*
_output_shapes

:
(*
dtype02"
 sequential/lstm/ReadVariableOp_3?
%sequential/lstm/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential/lstm/strided_slice_6/stack?
'sequential/lstm/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'sequential/lstm/strided_slice_6/stack_1?
'sequential/lstm/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'sequential/lstm/strided_slice_6/stack_2?
sequential/lstm/strided_slice_6StridedSlice(sequential/lstm/ReadVariableOp_3:value:0.sequential/lstm/strided_slice_6/stack:output:00sequential/lstm/strided_slice_6/stack_1:output:00sequential/lstm/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2!
sequential/lstm/strided_slice_6?
sequential/lstm/MatMul_7MatMulsequential/lstm/zeros:output:0(sequential/lstm/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/MatMul_7?
sequential/lstm/add_6AddV2"sequential/lstm/BiasAdd_3:output:0"sequential/lstm/MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/add_6w
sequential/lstm/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
sequential/lstm/Const_6w
sequential/lstm/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/lstm/Const_7?
sequential/lstm/Mul_4Mulsequential/lstm/add_6:z:0 sequential/lstm/Const_6:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/Mul_4?
sequential/lstm/Add_7Addsequential/lstm/Mul_4:z:0 sequential/lstm/Const_7:output:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/Add_7?
)sequential/lstm/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)sequential/lstm/clip_by_value_2/Minimum/y?
'sequential/lstm/clip_by_value_2/MinimumMinimumsequential/lstm/Add_7:z:02sequential/lstm/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2)
'sequential/lstm/clip_by_value_2/Minimum?
!sequential/lstm/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/lstm/clip_by_value_2/y?
sequential/lstm/clip_by_value_2Maximum+sequential/lstm/clip_by_value_2/Minimum:z:0*sequential/lstm/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2!
sequential/lstm/clip_by_value_2?
sequential/lstm/Tanh_1Tanhsequential/lstm/add_5:z:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/Tanh_1?
sequential/lstm/mul_5Mul#sequential/lstm/clip_by_value_2:z:0sequential/lstm/Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
sequential/lstm/mul_5?
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2/
-sequential/lstm/TensorArrayV2_1/element_shape?
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
sequential/lstm/TensorArrayV2_1n
sequential/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/lstm/time?
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(sequential/lstm/while/maximum_iterations?
"sequential/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential/lstm/while/loop_counter?
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-sequential_lstm_split_readvariableop_resource/sequential_lstm_split_1_readvariableop_resource'sequential_lstm_readvariableop_resource!^sequential/lstm/ReadVariableOp_3%^sequential/lstm/split/ReadVariableOp'^sequential/lstm/split_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *+
body#R!
sequential_lstm_while_body_7556*+
cond#R!
sequential_lstm_while_cond_7555*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
sequential/lstm/while?
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2B
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape?
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:??????????
*
element_dtype024
2sequential/lstm/TensorArrayV2Stack/TensorListStack?
%sequential/lstm/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2'
%sequential/lstm/strided_slice_7/stack?
'sequential/lstm/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential/lstm/strided_slice_7/stack_1?
'sequential/lstm/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_7/stack_2?
sequential/lstm/strided_slice_7StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_7/stack:output:00sequential/lstm/strided_slice_7/stack_1:output:00sequential/lstm/strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2!
sequential/lstm/strided_slice_7?
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 sequential/lstm/transpose_1/perm?
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:??????????
2
sequential/lstm/transpose_1?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul(sequential/lstm/strided_slice_7:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
IdentityIdentity!sequential/dense/BiasAdd:output:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp^sequential/lstm/ReadVariableOp!^sequential/lstm/ReadVariableOp_1!^sequential/lstm/ReadVariableOp_2!^sequential/lstm/ReadVariableOp_3%^sequential/lstm/split/ReadVariableOp'^sequential/lstm/split_1/ReadVariableOp^sequential/lstm/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2@
sequential/lstm/ReadVariableOpsequential/lstm/ReadVariableOp2D
 sequential/lstm/ReadVariableOp_1 sequential/lstm/ReadVariableOp_12D
 sequential/lstm/ReadVariableOp_2 sequential/lstm/ReadVariableOp_22D
 sequential/lstm/ReadVariableOp_3 sequential/lstm/ReadVariableOp_32L
$sequential/lstm/split/ReadVariableOp$sequential/lstm/split/ReadVariableOp2P
&sequential/lstm/split_1/ReadVariableOp&sequential/lstm/split_1/ReadVariableOp2.
sequential/lstm/whilesequential/lstm/while:* &
$
_user_specified_name
lstm_input
??
?
D__inference_sequential_layer_call_and_return_conditional_losses_9341

inputs&
"lstm_split_readvariableop_resource(
$lstm_split_1_readvariableop_resource 
lstm_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?lstm/ReadVariableOp?lstm/ReadVariableOp_1?lstm/ReadVariableOp_2?lstm/ReadVariableOp_3?lstm/split/ReadVariableOp?lstm/split_1/ReadVariableOp?
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack?
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1?
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros/mul/y?
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros/packed/1?
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const?

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros_1/mul/y?
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros_1/Less/y?
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros_1/packed/1?
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm?
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:??????????2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1?
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack?
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1?
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2?
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1?
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm/TensorArrayV2/element_shape?
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2?
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor?
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack?
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1?
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2?
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm/strided_slice_2Z

lstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

lstm/Constn
lstm/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/split/split_dim?
lstm/split/ReadVariableOpReadVariableOp"lstm_split_readvariableop_resource*
_output_shapes

:(*
dtype02
lstm/split/ReadVariableOp?

lstm/splitSplitlstm/split/split_dim:output:0!lstm/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2

lstm/split?
lstm/MatMulMatMullstm/strided_slice_2:output:0lstm/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul?
lstm/MatMul_1MatMullstm/strided_slice_2:output:0lstm/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_1?
lstm/MatMul_2MatMullstm/strided_slice_2:output:0lstm/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_2?
lstm/MatMul_3MatMullstm/strided_slice_2:output:0lstm/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_3^
lstm/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/Const_1r
lstm/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/split_1/split_dim?
lstm/split_1/ReadVariableOpReadVariableOp$lstm_split_1_readvariableop_resource*
_output_shapes
:(*
dtype02
lstm/split_1/ReadVariableOp?
lstm/split_1Splitlstm/split_1/split_dim:output:0#lstm/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2
lstm/split_1?
lstm/BiasAddBiasAddlstm/MatMul:product:0lstm/split_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm/BiasAdd?
lstm/BiasAdd_1BiasAddlstm/MatMul_1:product:0lstm/split_1:output:1*
T0*'
_output_shapes
:?????????
2
lstm/BiasAdd_1?
lstm/BiasAdd_2BiasAddlstm/MatMul_2:product:0lstm/split_1:output:2*
T0*'
_output_shapes
:?????????
2
lstm/BiasAdd_2?
lstm/BiasAdd_3BiasAddlstm/MatMul_3:product:0lstm/split_1:output:3*
T0*'
_output_shapes
:?????????
2
lstm/BiasAdd_3?
lstm/ReadVariableOpReadVariableOplstm_readvariableop_resource*
_output_shapes

:
(*
dtype02
lstm/ReadVariableOp?
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm/strided_slice_3/stack?
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
lstm/strided_slice_3/stack_1?
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lstm/strided_slice_3/stack_2?
lstm/strided_slice_3StridedSlicelstm/ReadVariableOp:value:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
lstm/strided_slice_3?
lstm/MatMul_4MatMullstm/zeros:output:0lstm/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_4
lstm/addAddV2lstm/BiasAdd:output:0lstm/MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2

lstm/adda
lstm/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm/Const_2a
lstm/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/Const_3r
lstm/MulMullstm/add:z:0lstm/Const_2:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Mulv

lstm/Add_1Addlstm/Mul:z:0lstm/Const_3:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Add_1?
lstm/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm/clip_by_value/Minimum/y?
lstm/clip_by_value/MinimumMinimumlstm/Add_1:z:0%lstm/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value/Minimumq
lstm/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/clip_by_value/y?
lstm/clip_by_valueMaximumlstm/clip_by_value/Minimum:z:0lstm/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value?
lstm/ReadVariableOp_1ReadVariableOplstm_readvariableop_resource^lstm/ReadVariableOp*
_output_shapes

:
(*
dtype02
lstm/ReadVariableOp_1?
lstm/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
lstm/strided_slice_4/stack?
lstm/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
lstm/strided_slice_4/stack_1?
lstm/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lstm/strided_slice_4/stack_2?
lstm/strided_slice_4StridedSlicelstm/ReadVariableOp_1:value:0#lstm/strided_slice_4/stack:output:0%lstm/strided_slice_4/stack_1:output:0%lstm/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
lstm/strided_slice_4?
lstm/MatMul_5MatMullstm/zeros:output:0lstm/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_5?

lstm/add_2AddV2lstm/BiasAdd_1:output:0lstm/MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2

lstm/add_2a
lstm/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm/Const_4a
lstm/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/Const_5x

lstm/Mul_1Mullstm/add_2:z:0lstm/Const_4:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Mul_1x

lstm/Add_3Addlstm/Mul_1:z:0lstm/Const_5:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Add_3?
lstm/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm/clip_by_value_1/Minimum/y?
lstm/clip_by_value_1/MinimumMinimumlstm/Add_3:z:0'lstm/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value_1/Minimumu
lstm/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/clip_by_value_1/y?
lstm/clip_by_value_1Maximum lstm/clip_by_value_1/Minimum:z:0lstm/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value_1?

lstm/mul_2Mullstm/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????
2

lstm/mul_2?
lstm/ReadVariableOp_2ReadVariableOplstm_readvariableop_resource^lstm/ReadVariableOp_1*
_output_shapes

:
(*
dtype02
lstm/ReadVariableOp_2?
lstm/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
lstm/strided_slice_5/stack?
lstm/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
lstm/strided_slice_5/stack_1?
lstm/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lstm/strided_slice_5/stack_2?
lstm/strided_slice_5StridedSlicelstm/ReadVariableOp_2:value:0#lstm/strided_slice_5/stack:output:0%lstm/strided_slice_5/stack_1:output:0%lstm/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
lstm/strided_slice_5?
lstm/MatMul_6MatMullstm/zeros:output:0lstm/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_6?

lstm/add_4AddV2lstm/BiasAdd_2:output:0lstm/MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2

lstm/add_4`
	lstm/TanhTanhlstm/add_4:z:0*
T0*'
_output_shapes
:?????????
2
	lstm/Tanhx

lstm/mul_3Mullstm/clip_by_value:z:0lstm/Tanh:y:0*
T0*'
_output_shapes
:?????????
2

lstm/mul_3s

lstm/add_5AddV2lstm/mul_2:z:0lstm/mul_3:z:0*
T0*'
_output_shapes
:?????????
2

lstm/add_5?
lstm/ReadVariableOp_3ReadVariableOplstm_readvariableop_resource^lstm/ReadVariableOp_2*
_output_shapes

:
(*
dtype02
lstm/ReadVariableOp_3?
lstm/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
lstm/strided_slice_6/stack?
lstm/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
lstm/strided_slice_6/stack_1?
lstm/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lstm/strided_slice_6/stack_2?
lstm/strided_slice_6StridedSlicelstm/ReadVariableOp_3:value:0#lstm/strided_slice_6/stack:output:0%lstm/strided_slice_6/stack_1:output:0%lstm/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
lstm/strided_slice_6?
lstm/MatMul_7MatMullstm/zeros:output:0lstm/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_7?

lstm/add_6AddV2lstm/BiasAdd_3:output:0lstm/MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2

lstm/add_6a
lstm/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm/Const_6a
lstm/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/Const_7x

lstm/Mul_4Mullstm/add_6:z:0lstm/Const_6:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Mul_4x

lstm/Add_7Addlstm/Mul_4:z:0lstm/Const_7:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Add_7?
lstm/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm/clip_by_value_2/Minimum/y?
lstm/clip_by_value_2/MinimumMinimumlstm/Add_7:z:0'lstm/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value_2/Minimumu
lstm/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/clip_by_value_2/y?
lstm/clip_by_value_2Maximum lstm/clip_by_value_2/Minimum:z:0lstm/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value_2d
lstm/Tanh_1Tanhlstm/add_5:z:0*
T0*'
_output_shapes
:?????????
2
lstm/Tanh_1|

lstm/mul_5Mullstm/clip_by_value_2:z:0lstm/Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2

lstm/mul_5?
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2$
"lstm/TensorArrayV2_1/element_shape?
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time?
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter?

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0"lstm_split_readvariableop_resource$lstm_split_1_readvariableop_resourcelstm_readvariableop_resource^lstm/ReadVariableOp_3^lstm/split/ReadVariableOp^lstm/split_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : * 
bodyR
lstm_while_body_9191* 
condR
lstm_while_cond_9190*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2

lstm/while?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:??????????
*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack?
lstm/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm/strided_slice_7/stack?
lstm/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_7/stack_1?
lstm/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_7/stack_2?
lstm/strided_slice_7StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_7/stack:output:0%lstm/strided_slice_7/stack_1:output:0%lstm/strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
lstm/strided_slice_7?
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:??????????
2
lstm/transpose_1?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMullstm/strided_slice_7:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const?
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^lstm/ReadVariableOp^lstm/ReadVariableOp_1^lstm/ReadVariableOp_2^lstm/ReadVariableOp_3^lstm/split/ReadVariableOp^lstm/split_1/ReadVariableOp^lstm/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2*
lstm/ReadVariableOplstm/ReadVariableOp2.
lstm/ReadVariableOp_1lstm/ReadVariableOp_12.
lstm/ReadVariableOp_2lstm/ReadVariableOp_22.
lstm/ReadVariableOp_3lstm/ReadVariableOp_326
lstm/split/ReadVariableOplstm/split/ReadVariableOp2:
lstm/split_1/ReadVariableOplstm/split_1/ReadVariableOp2

lstm/while
lstm/while:& "
 
_user_specified_nameinputs
?V
?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_10863

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????
2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicer
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_1t
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1f
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2t
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3t
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????
:?????????
:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
?j
?
while_body_8525
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
split_readvariableop_resource_0%
!split_1_readvariableop_resource_0
readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource_0*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
split?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMul?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2?
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOp!split_1_readvariableop_resource_0*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicew
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2y
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3y
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource_0^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4y
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_8/yW
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: 2
add_8T
add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_9/y^
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: 2
add_9?
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"4
readvariableop_resourcereadvariableop_resource_0"D
split_1_readvariableop_resource!split_1_readvariableop_resource_0"@
split_readvariableop_resourcesplit_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp
?
?
lstm_while_cond_9190
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_11
-lstm_while_cond_9190___redundant_placeholder01
-lstm_while_cond_9190___redundant_placeholder11
-lstm_while_cond_9190___redundant_placeholder21
-lstm_while_cond_9190___redundant_placeholder3
identity
]
LessLessplaceholderless_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
?
?
lstm_while_cond_9466
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_strided_slice_11
-lstm_while_cond_9466___redundant_placeholder01
-lstm_while_cond_9466___redundant_placeholder11
-lstm_while_cond_9466___redundant_placeholder21
-lstm_while_cond_9466___redundant_placeholder3
identity
]
LessLessplaceholderless_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
?
?
while_cond_8524
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1,
(while_cond_8524___redundant_placeholder0,
(while_cond_8524___redundant_placeholder1,
(while_cond_8524___redundant_placeholder2,
(while_cond_8524___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
?
?
sequential_lstm_while_cond_7555&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3(
$less_sequential_lstm_strided_slice_1<
8sequential_lstm_while_cond_7555___redundant_placeholder0<
8sequential_lstm_while_cond_7555___redundant_placeholder1<
8sequential_lstm_while_cond_7555___redundant_placeholder2<
8sequential_lstm_while_cond_7555___redundant_placeholder3
identity
h
LessLessplaceholder$less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
ˍ
?
>__inference_lstm_layer_call_and_return_conditional_losses_9907
inputs_0!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitv
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMulz
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1z
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2z
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3z
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4z
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_5z
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_6z
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcereadvariableop_resource^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *
bodyR
while_body_9764*
condR
while_cond_9763*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_7y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0
?
?
$__inference_lstm_layer_call_fn_10184
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_82672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
?V
?
C__inference_lstm_cell_layer_call_and_return_conditional_losses_7928

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????
2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicep
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_1r
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1f
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2r
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3r
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????
:?????????
:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates
?V
?
C__inference_lstm_cell_layer_call_and_return_conditional_losses_7836

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:?????????
2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicep
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_1r
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1f
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2r
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3r
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????
:?????????
:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates
?
?
while_cond_8200
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1,
(while_cond_8200___redundant_placeholder0,
(while_cond_8200___redundant_placeholder1,
(while_cond_8200___redundant_placeholder2,
(while_cond_8200___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
?
?
while_cond_10032
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_10032___redundant_placeholder0-
)while_cond_10032___redundant_placeholder1-
)while_cond_10032___redundant_placeholder2-
)while_cond_10032___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
?
?
while_cond_10586
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1-
)while_cond_10586___redundant_placeholder0-
)while_cond_10586___redundant_placeholder1-
)while_cond_10586___redundant_placeholder2-
)while_cond_10586___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
??
?
D__inference_sequential_layer_call_and_return_conditional_losses_9617

inputs&
"lstm_split_readvariableop_resource(
$lstm_split_1_readvariableop_resource 
lstm_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?lstm/ReadVariableOp?lstm/ReadVariableOp_1?lstm/ReadVariableOp_2?lstm/ReadVariableOp_3?lstm/split/ReadVariableOp?lstm/split_1/ReadVariableOp?
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack?
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1?
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2?
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros/mul/y?
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros/packed/1?
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const?

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros_1/mul/y?
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm/zeros_1/Less/y?
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
lstm/zeros_1/packed/1?
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const?
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm?
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:??????????2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1?
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack?
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1?
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2?
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1?
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm/TensorArrayV2/element_shape?
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2?
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape?
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor?
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack?
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1?
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2?
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm/strided_slice_2Z

lstm/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

lstm/Constn
lstm/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/split/split_dim?
lstm/split/ReadVariableOpReadVariableOp"lstm_split_readvariableop_resource*
_output_shapes

:(*
dtype02
lstm/split/ReadVariableOp?

lstm/splitSplitlstm/split/split_dim:output:0!lstm/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2

lstm/split?
lstm/MatMulMatMullstm/strided_slice_2:output:0lstm/split:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul?
lstm/MatMul_1MatMullstm/strided_slice_2:output:0lstm/split:output:1*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_1?
lstm/MatMul_2MatMullstm/strided_slice_2:output:0lstm/split:output:2*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_2?
lstm/MatMul_3MatMullstm/strided_slice_2:output:0lstm/split:output:3*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_3^
lstm/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/Const_1r
lstm/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/split_1/split_dim?
lstm/split_1/ReadVariableOpReadVariableOp$lstm_split_1_readvariableop_resource*
_output_shapes
:(*
dtype02
lstm/split_1/ReadVariableOp?
lstm/split_1Splitlstm/split_1/split_dim:output:0#lstm/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2
lstm/split_1?
lstm/BiasAddBiasAddlstm/MatMul:product:0lstm/split_1:output:0*
T0*'
_output_shapes
:?????????
2
lstm/BiasAdd?
lstm/BiasAdd_1BiasAddlstm/MatMul_1:product:0lstm/split_1:output:1*
T0*'
_output_shapes
:?????????
2
lstm/BiasAdd_1?
lstm/BiasAdd_2BiasAddlstm/MatMul_2:product:0lstm/split_1:output:2*
T0*'
_output_shapes
:?????????
2
lstm/BiasAdd_2?
lstm/BiasAdd_3BiasAddlstm/MatMul_3:product:0lstm/split_1:output:3*
T0*'
_output_shapes
:?????????
2
lstm/BiasAdd_3?
lstm/ReadVariableOpReadVariableOplstm_readvariableop_resource*
_output_shapes

:
(*
dtype02
lstm/ReadVariableOp?
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm/strided_slice_3/stack?
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
lstm/strided_slice_3/stack_1?
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lstm/strided_slice_3/stack_2?
lstm/strided_slice_3StridedSlicelstm/ReadVariableOp:value:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
lstm/strided_slice_3?
lstm/MatMul_4MatMullstm/zeros:output:0lstm/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_4
lstm/addAddV2lstm/BiasAdd:output:0lstm/MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2

lstm/adda
lstm/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm/Const_2a
lstm/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/Const_3r
lstm/MulMullstm/add:z:0lstm/Const_2:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Mulv

lstm/Add_1Addlstm/Mul:z:0lstm/Const_3:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Add_1?
lstm/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
lstm/clip_by_value/Minimum/y?
lstm/clip_by_value/MinimumMinimumlstm/Add_1:z:0%lstm/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value/Minimumq
lstm/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/clip_by_value/y?
lstm/clip_by_valueMaximumlstm/clip_by_value/Minimum:z:0lstm/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value?
lstm/ReadVariableOp_1ReadVariableOplstm_readvariableop_resource^lstm/ReadVariableOp*
_output_shapes

:
(*
dtype02
lstm/ReadVariableOp_1?
lstm/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
lstm/strided_slice_4/stack?
lstm/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
lstm/strided_slice_4/stack_1?
lstm/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lstm/strided_slice_4/stack_2?
lstm/strided_slice_4StridedSlicelstm/ReadVariableOp_1:value:0#lstm/strided_slice_4/stack:output:0%lstm/strided_slice_4/stack_1:output:0%lstm/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
lstm/strided_slice_4?
lstm/MatMul_5MatMullstm/zeros:output:0lstm/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_5?

lstm/add_2AddV2lstm/BiasAdd_1:output:0lstm/MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2

lstm/add_2a
lstm/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm/Const_4a
lstm/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/Const_5x

lstm/Mul_1Mullstm/add_2:z:0lstm/Const_4:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Mul_1x

lstm/Add_3Addlstm/Mul_1:z:0lstm/Const_5:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Add_3?
lstm/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm/clip_by_value_1/Minimum/y?
lstm/clip_by_value_1/MinimumMinimumlstm/Add_3:z:0'lstm/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value_1/Minimumu
lstm/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/clip_by_value_1/y?
lstm/clip_by_value_1Maximum lstm/clip_by_value_1/Minimum:z:0lstm/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value_1?

lstm/mul_2Mullstm/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*'
_output_shapes
:?????????
2

lstm/mul_2?
lstm/ReadVariableOp_2ReadVariableOplstm_readvariableop_resource^lstm/ReadVariableOp_1*
_output_shapes

:
(*
dtype02
lstm/ReadVariableOp_2?
lstm/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
lstm/strided_slice_5/stack?
lstm/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
lstm/strided_slice_5/stack_1?
lstm/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lstm/strided_slice_5/stack_2?
lstm/strided_slice_5StridedSlicelstm/ReadVariableOp_2:value:0#lstm/strided_slice_5/stack:output:0%lstm/strided_slice_5/stack_1:output:0%lstm/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
lstm/strided_slice_5?
lstm/MatMul_6MatMullstm/zeros:output:0lstm/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_6?

lstm/add_4AddV2lstm/BiasAdd_2:output:0lstm/MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2

lstm/add_4`
	lstm/TanhTanhlstm/add_4:z:0*
T0*'
_output_shapes
:?????????
2
	lstm/Tanhx

lstm/mul_3Mullstm/clip_by_value:z:0lstm/Tanh:y:0*
T0*'
_output_shapes
:?????????
2

lstm/mul_3s

lstm/add_5AddV2lstm/mul_2:z:0lstm/mul_3:z:0*
T0*'
_output_shapes
:?????????
2

lstm/add_5?
lstm/ReadVariableOp_3ReadVariableOplstm_readvariableop_resource^lstm/ReadVariableOp_2*
_output_shapes

:
(*
dtype02
lstm/ReadVariableOp_3?
lstm/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
lstm/strided_slice_6/stack?
lstm/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
lstm/strided_slice_6/stack_1?
lstm/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
lstm/strided_slice_6/stack_2?
lstm/strided_slice_6StridedSlicelstm/ReadVariableOp_3:value:0#lstm/strided_slice_6/stack:output:0%lstm/strided_slice_6/stack_1:output:0%lstm/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
lstm/strided_slice_6?
lstm/MatMul_7MatMullstm/zeros:output:0lstm/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????
2
lstm/MatMul_7?

lstm/add_6AddV2lstm/BiasAdd_3:output:0lstm/MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2

lstm/add_6a
lstm/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm/Const_6a
lstm/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/Const_7x

lstm/Mul_4Mullstm/add_6:z:0lstm/Const_6:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Mul_4x

lstm/Add_7Addlstm/Mul_4:z:0lstm/Const_7:output:0*
T0*'
_output_shapes
:?????????
2

lstm/Add_7?
lstm/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
lstm/clip_by_value_2/Minimum/y?
lstm/clip_by_value_2/MinimumMinimumlstm/Add_7:z:0'lstm/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value_2/Minimumu
lstm/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/clip_by_value_2/y?
lstm/clip_by_value_2Maximum lstm/clip_by_value_2/Minimum:z:0lstm/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
lstm/clip_by_value_2d
lstm/Tanh_1Tanhlstm/add_5:z:0*
T0*'
_output_shapes
:?????????
2
lstm/Tanh_1|

lstm/mul_5Mullstm/clip_by_value_2:z:0lstm/Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2

lstm/mul_5?
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2$
"lstm/TensorArrayV2_1/element_shape?
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time?
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter?

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0"lstm_split_readvariableop_resource$lstm_split_1_readvariableop_resourcelstm_readvariableop_resource^lstm/ReadVariableOp_3^lstm/split/ReadVariableOp^lstm/split_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : * 
bodyR
lstm_while_body_9467* 
condR
lstm_while_cond_9466*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2

lstm/while?
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape?
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:??????????
*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack?
lstm/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm/strided_slice_7/stack?
lstm/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_7/stack_1?
lstm/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_7/stack_2?
lstm/strided_slice_7StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_7/stack:output:0%lstm/strided_slice_7/stack_1:output:0%lstm/strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
lstm/strided_slice_7?
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm?
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*+
_output_shapes
:??????????
2
lstm/transpose_1?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMullstm/strided_slice_7:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Const?
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^lstm/ReadVariableOp^lstm/ReadVariableOp_1^lstm/ReadVariableOp_2^lstm/ReadVariableOp_3^lstm/split/ReadVariableOp^lstm/split_1/ReadVariableOp^lstm/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2*
lstm/ReadVariableOplstm/ReadVariableOp2.
lstm/ReadVariableOp_1lstm/ReadVariableOp_12.
lstm/ReadVariableOp_2lstm/ReadVariableOp_22.
lstm/ReadVariableOp_3lstm/ReadVariableOp_326
lstm/split/ReadVariableOplstm/split/ReadVariableOp2:
lstm/split_1/ReadVariableOplstm/split_1/ReadVariableOp2

lstm/while
lstm/while:& "
 
_user_specified_nameinputs
?
+
__inference_loss_fn_0_10770
identity?
dense/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/Constj
IdentityIdentity'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
)__inference_sequential_layer_call_fn_9637

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_90362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:??????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
while_cond_8323
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1,
(while_cond_8323___redundant_placeholder0,
(while_cond_8323___redundant_placeholder1,
(while_cond_8323___redundant_placeholder2,
(while_cond_8323___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :?????????
:?????????
: ::::
?l
?
sequential_lstm_while_body_7556&
"sequential_lstm_while_loop_counter,
(sequential_lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3%
!sequential_lstm_strided_slice_1_0a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0#
split_readvariableop_resource_0%
!split_1_readvariableop_resource_0
readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5#
sequential_lstm_strided_slice_1_
[tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItem]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource_0*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
split?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMul?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2?
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOp!split_1_readvariableop_resource_0*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicew
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_1y
MatMul_5MatMulplaceholder_2strided_slice_1:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2y
MatMul_6MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource_0^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3y
MatMul_7MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_8/yW
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: 2
add_8T
add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_9/yn
add_9AddV2"sequential_lstm_while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: 2
add_9?
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity?

Identity_1Identity(sequential_lstm_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"4
readvariableop_resourcereadvariableop_resource_0"D
sequential_lstm_strided_slice_1!sequential_lstm_strided_slice_1_0"D
split_1_readvariableop_resource!split_1_readvariableop_resource_0"@
split_readvariableop_resourcesplit_readvariableop_resource_0"?
[tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor]tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp
?j
?
while_body_9764
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
split_readvariableop_resource_0%
!split_1_readvariableop_resource_0
readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource_0*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
split?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMul?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2?
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOp!split_1_readvariableop_resource_0*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicew
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2y
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3y
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource_0^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4y
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_8/yW
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: 2
add_8T
add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_9/y^
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: 2
add_9?
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"4
readvariableop_resourcereadvariableop_resource_0"D
split_1_readvariableop_resource!split_1_readvariableop_resource_0"@
split_readvariableop_resourcesplit_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp
?j
?
lstm_while_body_9191
lstm_while_loop_counter!
lstm_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_strided_slice_1_0V
Rtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0#
split_readvariableop_resource_0%
!split_1_readvariableop_resource_0
readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_strided_slice_1T
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource_0*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
split?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMul?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2?
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOp!split_1_readvariableop_resource_0*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicew
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_1y
MatMul_5MatMulplaceholder_2strided_slice_1:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2y
MatMul_6MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource_0^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3y
MatMul_7MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_8/yW
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: 2
add_8T
add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_9/yc
add_9AddV2lstm_while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: 2
add_9?
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitylstm_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0".
lstm_strided_slice_1lstm_strided_slice_1_0"4
readvariableop_resourcereadvariableop_resource_0"D
split_1_readvariableop_resource!split_1_readvariableop_resource_0"@
split_readvariableop_resourcesplit_readvariableop_resource_0"?
Ptensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensorRtensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp
?
?
)__inference_lstm_cell_layer_call_fn_10983

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*M
_output_shapes;
9:?????????
:?????????
:?????????
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_79282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????
:?????????
:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1
?j
?
while_body_10587
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
split_readvariableop_resource_0%
!split_1_readvariableop_resource_0
readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource_0*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
split?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMul?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2?
MatMul_3MatMul*TensorArrayV2Read/TensorListGetItem:item:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOp!split_1_readvariableop_resource_0*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:
(*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slicew
MatMul_4MatMulplaceholder_2strided_slice:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_2y
MatMul_5MatMulplaceholder_2strided_slice_2:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_3*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3y
MatMul_6MatMulplaceholder_2strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource_0^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4y
MatMul_7MatMulplaceholder_2strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_8/yW
add_8AddV2placeholderadd_8/y:output:0*
T0*
_output_shapes
: 2
add_8T
add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_9/y^
add_9AddV2while_loop_counteradd_9/y:output:0*
T0*
_output_shapes
: 2
add_9?
IdentityIdentity	add_9:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_8:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_4?

Identity_5Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"4
readvariableop_resourcereadvariableop_resource_0"D
split_1_readvariableop_resource!split_1_readvariableop_resource_0"@
split_readvariableop_resourcesplit_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????
:?????????
: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp
΍
?
?__inference_lstm_layer_call_and_return_conditional_losses_10176
inputs_0!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????
2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????
2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:(*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(:
:
:
:
*
	num_split2
splitv
MatMulMatMulstrided_slice_2:output:0split:output:0*
T0*'
_output_shapes
:?????????
2
MatMulz
MatMul_1MatMulstrided_slice_2:output:0split:output:1*
T0*'
_output_shapes
:?????????
2

MatMul_1z
MatMul_2MatMulstrided_slice_2:output:0split:output:2*
T0*'
_output_shapes
:?????????
2

MatMul_2z
MatMul_3MatMulstrided_slice_2:output:0split:output:3*
T0*'
_output_shapes
:?????????
2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:(*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:
:
:
:
*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:?????????
2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:?????????
2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:?????????
2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:?????????
2
	BiasAdd_3x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:
(*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_3z
MatMul_4MatMulzeros:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????
2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3^
MulMuladd:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????
2
Mulb
Add_1AddMul:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????
2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:
(*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_4z
MatMul_5MatMulzeros:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_5q
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????
2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5d
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*'
_output_shapes
:?????????
2
Mul_1d
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*'
_output_shapes
:?????????
2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:?????????
2
mul_2?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:
(*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_5z
MatMul_6MatMulzeros:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_6q
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:?????????
2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????
2
Tanhd
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????
2
mul_3_
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????
2
add_5?
ReadVariableOp_3ReadVariableOpreadvariableop_resource^ReadVariableOp_2*
_output_shapes

:
(*
dtype02
ReadVariableOp_3
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:

*

begin_mask*
end_mask2
strided_slice_6z
MatMul_7MatMulzeros:output:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????
2

MatMul_7q
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????
2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7d
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*'
_output_shapes
:?????????
2
Mul_4d
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*'
_output_shapes
:?????????
2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:?????????
2
clip_by_value_2U
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????
2
Tanh_1h
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????
2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resourcereadvariableop_resource^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????
:?????????
: : : : : *
bodyR
while_body_10033*
condR
while_cond_10032*K
output_shapes:
8: : : : :?????????
:?????????
: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????
   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????
*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????
*
shrink_axis_mask2
strided_slice_7y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????
2
transpose_1?
lstm/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_7:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E

lstm_input7
serving_default_lstm_input:0??????????9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
<__call__
=_default_save_signature
*>&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "batch_input_shape": [null, 63, 6], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 10, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 6], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "batch_input_shape": [null, 63, 6], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 10, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "lstm_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 63, 6], "config": {"batch_input_shape": [null, 63, 6], "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_input"}}
?


cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 63, 6], "config": {"name": "lstm", "trainable": true, "batch_input_shape": [null, 63, 6], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 10, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 6], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}}
?
iter

beta_1

beta_2
	decay
learning_ratem2m3m4m5m6v7v8v9v:v;"
	optimizer
'
C0"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
layer_regularization_losses
non_trainable_variables

 layers
regularization_losses
	variables
trainable_variables
!metrics
<__call__
=_default_save_signature
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
,
Dserving_default"
signature_map
?

kernel
recurrent_kernel
bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
E__call__
*F&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
&layer_regularization_losses
'non_trainable_variables

(layers
regularization_losses
	variables
trainable_variables
)metrics
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
:
2dense/kernel
:2
dense/bias
'
C0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
*layer_regularization_losses
+non_trainable_variables

,layers
regularization_losses
	variables
trainable_variables
-metrics
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:(2lstm/kernel
':%
(2lstm/recurrent_kernel
:(2	lstm/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
.layer_regularization_losses
/non_trainable_variables

0layers
"regularization_losses
#	variables
$trainable_variables
1metrics
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
#:!
2Adam/dense/kernel/m
:2Adam/dense/bias/m
": (2Adam/lstm/kernel/m
,:*
(2Adam/lstm/recurrent_kernel/m
:(2Adam/lstm/bias/m
#:!
2Adam/dense/kernel/v
:2Adam/dense/bias/v
": (2Adam/lstm/kernel/v
,:*
(2Adam/lstm/recurrent_kernel/v
:(2Adam/lstm/bias/v
?2?
)__inference_sequential_layer_call_fn_9627
)__inference_sequential_layer_call_fn_9044
)__inference_sequential_layer_call_fn_9021
)__inference_sequential_layer_call_fn_9637?
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
?2?
__inference__wrapped_model_7704?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *-?*
(?%

lstm_input??????????
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_9341
D__inference_sequential_layer_call_and_return_conditional_losses_8984
D__inference_sequential_layer_call_and_return_conditional_losses_9617
D__inference_sequential_layer_call_and_return_conditional_losses_8997?
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
?2?
$__inference_lstm_layer_call_fn_10192
$__inference_lstm_layer_call_fn_10738
$__inference_lstm_layer_call_fn_10746
$__inference_lstm_layer_call_fn_10184?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_lstm_layer_call_and_return_conditional_losses_9907
?__inference_lstm_layer_call_and_return_conditional_losses_10730
?__inference_lstm_layer_call_and_return_conditional_losses_10176
?__inference_lstm_layer_call_and_return_conditional_losses_10461?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_10765?
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
@__inference_dense_layer_call_and_return_conditional_losses_10758?
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
__inference_loss_fn_0_10770?
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
4B2
"__inference_signature_wrapper_9065
lstm_input
?2?
)__inference_lstm_cell_layer_call_fn_10983
)__inference_lstm_cell_layer_call_fn_10969?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_10955
D__inference_lstm_cell_layer_call_and_return_conditional_losses_10863?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_1_10988?
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
annotations? *? ?
__inference__wrapped_model_7704o7?4
-?*
(?%

lstm_input??????????
? "-?*
(
dense?
dense??????????
@__inference_dense_layer_call_and_return_conditional_losses_10758\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? x
%__inference_dense_layer_call_fn_10765O/?,
%?"
 ?
inputs?????????

? "??????????7
__inference_loss_fn_0_10770?

? 
? "? 7
__inference_loss_fn_1_10988?

? 
? "? ?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_10863???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????

"?
states/1?????????

p
? "s?p
i?f
?
0/0?????????

E?B
?
0/1/0?????????

?
0/1/1?????????

? ?
D__inference_lstm_cell_layer_call_and_return_conditional_losses_10955???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????

"?
states/1?????????

p 
? "s?p
i?f
?
0/0?????????

E?B
?
0/1/0?????????

?
0/1/1?????????

? ?
)__inference_lstm_cell_layer_call_fn_10969???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????

"?
states/1?????????

p
? "c?`
?
0?????????

A?>
?
1/0?????????

?
1/1?????????
?
)__inference_lstm_cell_layer_call_fn_10983???}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????

"?
states/1?????????

p 
? "c?`
?
0?????????

A?>
?
1/0?????????

?
1/1?????????
?
?__inference_lstm_layer_call_and_return_conditional_losses_10176}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????

? ?
?__inference_lstm_layer_call_and_return_conditional_losses_10461m??<
5?2
$?!
inputs??????????

 
p

 
? "%?"
?
0?????????

? ?
?__inference_lstm_layer_call_and_return_conditional_losses_10730m??<
5?2
$?!
inputs??????????

 
p 

 
? "%?"
?
0?????????

? ?
>__inference_lstm_layer_call_and_return_conditional_losses_9907}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????

? ?
$__inference_lstm_layer_call_fn_10184pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????
?
$__inference_lstm_layer_call_fn_10192pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????
?
$__inference_lstm_layer_call_fn_10738`??<
5?2
$?!
inputs??????????

 
p

 
? "??????????
?
$__inference_lstm_layer_call_fn_10746`??<
5?2
$?!
inputs??????????

 
p 

 
? "??????????
?
D__inference_sequential_layer_call_and_return_conditional_losses_8984o??<
5?2
(?%

lstm_input??????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_8997o??<
5?2
(?%

lstm_input??????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_9341k;?8
1?.
$?!
inputs??????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_9617k;?8
1?.
$?!
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
)__inference_sequential_layer_call_fn_9021b??<
5?2
(?%

lstm_input??????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_9044b??<
5?2
(?%

lstm_input??????????
p 

 
? "???????????
)__inference_sequential_layer_call_fn_9627^;?8
1?.
$?!
inputs??????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_9637^;?8
1?.
$?!
inputs??????????
p 

 
? "???????????
"__inference_signature_wrapper_9065}E?B
? 
;?8
6

lstm_input(?%

lstm_input??????????"-?*
(
dense?
dense?????????