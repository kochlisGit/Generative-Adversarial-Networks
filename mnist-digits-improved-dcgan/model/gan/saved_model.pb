Ў█-
юы
B
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

└
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Щ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%иЛ8"&
exponential_avg_factorfloat%  ђ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%═╠L>"
Ttype0:
2
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
delete_old_dirsbool(ѕ
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
dtypetypeѕ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.12v2.4.0-49-g85c8b2a817f8█Щ%
f
	Yogi/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Yogi/iter
_
Yogi/iter/Read/ReadVariableOpReadVariableOp	Yogi/iter*
_output_shapes
: *
dtype0	
j
Yogi/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameYogi/beta_1
c
Yogi/beta_1/Read/ReadVariableOpReadVariableOpYogi/beta_1*
_output_shapes
: *
dtype0
j
Yogi/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameYogi/beta_2
c
Yogi/beta_2/Read/ReadVariableOpReadVariableOpYogi/beta_2*
_output_shapes
: *
dtype0
h

Yogi/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Yogi/decay
a
Yogi/decay/Read/ReadVariableOpReadVariableOp
Yogi/decay*
_output_shapes
: *
dtype0
l
Yogi/epsilonVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameYogi/epsilon
e
 Yogi/epsilon/Read/ReadVariableOpReadVariableOpYogi/epsilon*
_output_shapes
: *
dtype0
њ
Yogi/l1_regularization_strengthVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Yogi/l1_regularization_strength
І
3Yogi/l1_regularization_strength/Read/ReadVariableOpReadVariableOpYogi/l1_regularization_strength*
_output_shapes
: *
dtype0
њ
Yogi/l2_regularization_strengthVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Yogi/l2_regularization_strength
І
3Yogi/l2_regularization_strength/Read/ReadVariableOpReadVariableOpYogi/l2_regularization_strength*
_output_shapes
: *
dtype0
x
Yogi/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameYogi/learning_rate
q
&Yogi/learning_rate/Read/ReadVariableOpReadVariableOpYogi/learning_rate*
_output_shapes
: *
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђb*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
ђђb*
dtype0
І
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђb**
shared_namebatch_normalization/gamma
ё
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:ђb*
dtype0
Ѕ
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђb*)
shared_namebatch_normalization/beta
ѓ
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:ђb*
dtype0
Ќ
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђb*0
shared_name!batch_normalization/moving_mean
љ
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:ђb*
dtype0
Ъ
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђb*4
shared_name%#batch_normalization/moving_variance
ў
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:ђb*
dtype0
ћ
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameconv2d_transpose/kernel
Ї
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*(
_output_shapes
:ђђ*
dtype0
Ј
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_1/gamma
ѕ
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_1/beta
є
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_1/moving_mean
ћ
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_1/moving_variance
ю
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:ђ*
dtype0
Ќ
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ**
shared_nameconv2d_transpose_1/kernel
љ
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*'
_output_shapes
:@ђ*
dtype0
ј
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
Є
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
ї
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Ё
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
џ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
Њ
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
б
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Џ
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
ќ
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameconv2d_transpose_2/kernel
Ј
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
:@*
dtype0
є
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
ј
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
Є
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
ї
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
Ё
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
џ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
Њ
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
б
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
Џ
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
Ѓ
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ* 
shared_nameconv2d_1/kernel
|
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*'
_output_shapes
:@ђ*
dtype0
Ј
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*,
shared_namebatch_normalization_4/gamma
ѕ
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:ђ*
dtype0
Ї
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*+
shared_namebatch_normalization_4/beta
є
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!batch_normalization_4/moving_mean
ћ
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*6
shared_name'%batch_normalization_4/moving_variance
ю
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:ђ*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ1*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	ђ1*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
j
Yogi/iter_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameYogi/iter_1
c
Yogi/iter_1/Read/ReadVariableOpReadVariableOpYogi/iter_1*
_output_shapes
: *
dtype0	
n
Yogi/beta_1_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameYogi/beta_1_1
g
!Yogi/beta_1_1/Read/ReadVariableOpReadVariableOpYogi/beta_1_1*
_output_shapes
: *
dtype0
n
Yogi/beta_2_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameYogi/beta_2_1
g
!Yogi/beta_2_1/Read/ReadVariableOpReadVariableOpYogi/beta_2_1*
_output_shapes
: *
dtype0
l
Yogi/decay_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameYogi/decay_1
e
 Yogi/decay_1/Read/ReadVariableOpReadVariableOpYogi/decay_1*
_output_shapes
: *
dtype0
p
Yogi/epsilon_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameYogi/epsilon_1
i
"Yogi/epsilon_1/Read/ReadVariableOpReadVariableOpYogi/epsilon_1*
_output_shapes
: *
dtype0
ќ
!Yogi/l1_regularization_strength_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Yogi/l1_regularization_strength_1
Ј
5Yogi/l1_regularization_strength_1/Read/ReadVariableOpReadVariableOp!Yogi/l1_regularization_strength_1*
_output_shapes
: *
dtype0
ќ
!Yogi/l2_regularization_strength_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Yogi/l2_regularization_strength_1
Ј
5Yogi/l2_regularization_strength_1/Read/ReadVariableOpReadVariableOp!Yogi/l2_regularization_strength_1*
_output_shapes
: *
dtype0
|
Yogi/learning_rate_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameYogi/learning_rate_1
u
(Yogi/learning_rate_1/Read/ReadVariableOpReadVariableOpYogi/learning_rate_1*
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
ё
Yogi/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђb*$
shared_nameYogi/dense/kernel/v
}
'Yogi/dense/kernel/v/Read/ReadVariableOpReadVariableOpYogi/dense/kernel/v* 
_output_shapes
:
ђђb*
dtype0
Ў
 Yogi/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђb*1
shared_name" Yogi/batch_normalization/gamma/v
њ
4Yogi/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Yogi/batch_normalization/gamma/v*
_output_shapes	
:ђb*
dtype0
Ќ
Yogi/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђb*0
shared_name!Yogi/batch_normalization/beta/v
љ
3Yogi/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpYogi/batch_normalization/beta/v*
_output_shapes	
:ђb*
dtype0
б
Yogi/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*/
shared_name Yogi/conv2d_transpose/kernel/v
Џ
2Yogi/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpYogi/conv2d_transpose/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ю
"Yogi/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Yogi/batch_normalization_1/gamma/v
ќ
6Yogi/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Yogi/batch_normalization_1/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Yogi/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Yogi/batch_normalization_1/beta/v
ћ
5Yogi/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Yogi/batch_normalization_1/beta/v*
_output_shapes	
:ђ*
dtype0
Ц
 Yogi/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*1
shared_name" Yogi/conv2d_transpose_1/kernel/v
ъ
4Yogi/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Yogi/conv2d_transpose_1/kernel/v*'
_output_shapes
:@ђ*
dtype0
ю
"Yogi/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Yogi/batch_normalization_2/gamma/v
Ћ
6Yogi/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Yogi/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
џ
!Yogi/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Yogi/batch_normalization_2/beta/v
Њ
5Yogi/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Yogi/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
ц
 Yogi/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Yogi/conv2d_transpose_2/kernel/v
Ю
4Yogi/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp Yogi/conv2d_transpose_2/kernel/v*&
_output_shapes
:@*
dtype0
ћ
Yogi/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Yogi/conv2d_transpose_2/bias/v
Ї
2Yogi/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpYogi/conv2d_transpose_2/bias/v*
_output_shapes
:*
dtype0
ё
Yogi/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђb*$
shared_nameYogi/dense/kernel/m
}
'Yogi/dense/kernel/m/Read/ReadVariableOpReadVariableOpYogi/dense/kernel/m* 
_output_shapes
:
ђђb*
dtype0
Ў
 Yogi/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђb*1
shared_name" Yogi/batch_normalization/gamma/m
њ
4Yogi/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Yogi/batch_normalization/gamma/m*
_output_shapes	
:ђb*
dtype0
Ќ
Yogi/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђb*0
shared_name!Yogi/batch_normalization/beta/m
љ
3Yogi/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpYogi/batch_normalization/beta/m*
_output_shapes	
:ђb*
dtype0
б
Yogi/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*/
shared_name Yogi/conv2d_transpose/kernel/m
Џ
2Yogi/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpYogi/conv2d_transpose/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ю
"Yogi/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Yogi/batch_normalization_1/gamma/m
ќ
6Yogi/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Yogi/batch_normalization_1/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Yogi/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Yogi/batch_normalization_1/beta/m
ћ
5Yogi/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Yogi/batch_normalization_1/beta/m*
_output_shapes	
:ђ*
dtype0
Ц
 Yogi/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*1
shared_name" Yogi/conv2d_transpose_1/kernel/m
ъ
4Yogi/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Yogi/conv2d_transpose_1/kernel/m*'
_output_shapes
:@ђ*
dtype0
ю
"Yogi/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Yogi/batch_normalization_2/gamma/m
Ћ
6Yogi/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Yogi/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
џ
!Yogi/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Yogi/batch_normalization_2/beta/m
Њ
5Yogi/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Yogi/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
ц
 Yogi/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Yogi/conv2d_transpose_2/kernel/m
Ю
4Yogi/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp Yogi/conv2d_transpose_2/kernel/m*&
_output_shapes
:@*
dtype0
ћ
Yogi/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Yogi/conv2d_transpose_2/bias/m
Ї
2Yogi/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpYogi/conv2d_transpose_2/bias/m*
_output_shapes
:*
dtype0
ї
Yogi/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameYogi/conv2d/kernel/v
Ё
(Yogi/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpYogi/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
ю
"Yogi/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Yogi/batch_normalization_3/gamma/v
Ћ
6Yogi/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Yogi/batch_normalization_3/gamma/v*
_output_shapes
:@*
dtype0
џ
!Yogi/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Yogi/batch_normalization_3/beta/v
Њ
5Yogi/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Yogi/batch_normalization_3/beta/v*
_output_shapes
:@*
dtype0
Љ
Yogi/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*'
shared_nameYogi/conv2d_1/kernel/v
і
*Yogi/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpYogi/conv2d_1/kernel/v*'
_output_shapes
:@ђ*
dtype0
Ю
"Yogi/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Yogi/batch_normalization_4/gamma/v
ќ
6Yogi/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Yogi/batch_normalization_4/gamma/v*
_output_shapes	
:ђ*
dtype0
Џ
!Yogi/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Yogi/batch_normalization_4/beta/v
ћ
5Yogi/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Yogi/batch_normalization_4/beta/v*
_output_shapes	
:ђ*
dtype0
Є
Yogi/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ1*&
shared_nameYogi/dense_1/kernel/v
ђ
)Yogi/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpYogi/dense_1/kernel/v*
_output_shapes
:	ђ1*
dtype0
~
Yogi/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameYogi/dense_1/bias/v
w
'Yogi/dense_1/bias/v/Read/ReadVariableOpReadVariableOpYogi/dense_1/bias/v*
_output_shapes
:*
dtype0
ї
Yogi/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameYogi/conv2d/kernel/m
Ё
(Yogi/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpYogi/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
ю
"Yogi/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Yogi/batch_normalization_3/gamma/m
Ћ
6Yogi/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Yogi/batch_normalization_3/gamma/m*
_output_shapes
:@*
dtype0
џ
!Yogi/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Yogi/batch_normalization_3/beta/m
Њ
5Yogi/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Yogi/batch_normalization_3/beta/m*
_output_shapes
:@*
dtype0
Љ
Yogi/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*'
shared_nameYogi/conv2d_1/kernel/m
і
*Yogi/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpYogi/conv2d_1/kernel/m*'
_output_shapes
:@ђ*
dtype0
Ю
"Yogi/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*3
shared_name$"Yogi/batch_normalization_4/gamma/m
ќ
6Yogi/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Yogi/batch_normalization_4/gamma/m*
_output_shapes	
:ђ*
dtype0
Џ
!Yogi/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*2
shared_name#!Yogi/batch_normalization_4/beta/m
ћ
5Yogi/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Yogi/batch_normalization_4/beta/m*
_output_shapes	
:ђ*
dtype0
Є
Yogi/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ1*&
shared_nameYogi/dense_1/kernel/m
ђ
)Yogi/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpYogi/dense_1/kernel/m*
_output_shapes
:	ђ1*
dtype0
~
Yogi/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameYogi/dense_1/bias/m
w
'Yogi/dense_1/bias/m/Read/ReadVariableOpReadVariableOpYogi/dense_1/bias/m*
_output_shapes
:*
dtype0

NoOpNoOp
лб
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*іб
value АBчА BзА
┐
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
┤
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
layer_with_weights-4
layer-8
layer_with_weights-5
layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
з
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
 layer_with_weights-3
 layer-6
!layer-7
"layer-8
#layer-9
$layer_with_weights-4
$layer-10
%	optimizer
&	variables
'trainable_variables
(regularization_losses
)	keras_api
ж
*iter

+beta_1

,beta_2
	-decay
.epsilon
/l1_regularization_strength
0l2_regularization_strength
1learning_rate2v═3v╬4v¤7vл8vЛ9vм<vМ=vн>vНAvоBvО2mп3m┘4m┌7m█8m▄9mП<mя=m▀>mЯAmрBmР
я
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19
F20
G21
H22
I23
J24
K25
L26
M27
N28
N
20
31
42
73
84
95
<6
=7
>8
A9
B10
 
Г
Olayer_regularization_losses
	variables
trainable_variables
Player_metrics
Qmetrics
Rnon_trainable_variables

Slayers
regularization_losses
 
^

2kernel
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
Ќ
Xaxis
	3gamma
4beta
5moving_mean
6moving_variance
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
R
]	variables
^trainable_variables
_regularization_losses
`	keras_api
R
a	variables
btrainable_variables
cregularization_losses
d	keras_api
^

7kernel
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
Ќ
iaxis
	8gamma
9beta
:moving_mean
;moving_variance
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
R
n	variables
otrainable_variables
pregularization_losses
q	keras_api
R
r	variables
strainable_variables
tregularization_losses
u	keras_api
^

<kernel
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
Ќ
zaxis
	=gamma
>beta
?moving_mean
@moving_variance
{	variables
|trainable_variables
}regularization_losses
~	keras_api
U
	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
V
Ѓ	variables
ёtrainable_variables
Ёregularization_losses
є	keras_api
l

Akernel
Bbias
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
~
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
N
20
31
42
73
84
95
<6
=7
>8
A9
B10
 
▓
 Іlayer_regularization_losses
	variables
trainable_variables
їlayer_metrics
Їmetrics
јnon_trainable_variables
Јlayers
regularization_losses
V
љ	variables
Љtrainable_variables
њregularization_losses
Њ	keras_api
b

Ckernel
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
ю
	ўaxis
	Dgamma
Ebeta
Fmoving_mean
Gmoving_variance
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
V
Ю	variables
ъtrainable_variables
Ъregularization_losses
а	keras_api
V
А	variables
бtrainable_variables
Бregularization_losses
ц	keras_api
b

Hkernel
Ц	variables
дtrainable_variables
Дregularization_losses
е	keras_api
ю
	Еaxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
ф	variables
Фtrainable_variables
гregularization_losses
Г	keras_api
V
«	variables
»trainable_variables
░regularization_losses
▒	keras_api
V
▓	variables
│trainable_variables
┤regularization_losses
х	keras_api
V
Х	variables
иtrainable_variables
Иregularization_losses
╣	keras_api
l

Mkernel
Nbias
║	variables
╗trainable_variables
╝regularization_losses
й	keras_api
х
	Йiter
┐beta_1
└beta_2

┴decay
┬epsilon
├l1_regularization_strength
─l2_regularization_strength
┼learning_rateCvсDvСEvтHvТIvуJvУMvжNvЖCmвDmВEmьHmЬIm№Jm­MmыNmЫ
V
C0
D1
E2
F3
G4
H5
I6
J7
K8
L9
M10
N11
 
 
▓
 кlayer_regularization_losses
&	variables
'trainable_variables
Кlayer_metrics
╚metrics
╔non_trainable_variables
╩layers
(regularization_losses
HF
VARIABLE_VALUE	Yogi/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEYogi/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEYogi/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Yogi/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEYogi/epsilon,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEYogi/l1_regularization_strength?optimizer/l1_regularization_strength/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEYogi/l2_regularization_strength?optimizer/l2_regularization_strength/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEYogi/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/2/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_transpose/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_1/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_1/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_1/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_1/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_2/gamma'variables/11/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_2/beta'variables/12/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/13/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/14/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_2/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_2/bias'variables/16/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_3/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_3/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_3/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_3/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_1/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_4/gamma'variables/23/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_4/beta'variables/24/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_4/moving_mean'variables/25/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_4/moving_variance'variables/26/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/27/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/28/.ATTRIBUTES/VARIABLE_VALUE
 
 

╦0
є
50
61
:2
;3
?4
@5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17

0
1

20

20
 
▓
 ╠layer_regularization_losses
T	variables
Utrainable_variables
═layer_metrics
╬metrics
¤non_trainable_variables
лlayers
Vregularization_losses
 

30
41
52
63

30
41
 
▓
 Лlayer_regularization_losses
Y	variables
Ztrainable_variables
мlayer_metrics
Мmetrics
нnon_trainable_variables
Нlayers
[regularization_losses
 
 
 
▓
 оlayer_regularization_losses
]	variables
^trainable_variables
Оlayer_metrics
пmetrics
┘non_trainable_variables
┌layers
_regularization_losses
 
 
 
▓
 █layer_regularization_losses
a	variables
btrainable_variables
▄layer_metrics
Пmetrics
яnon_trainable_variables
▀layers
cregularization_losses

70

70
 
▓
 Яlayer_regularization_losses
e	variables
ftrainable_variables
рlayer_metrics
Рmetrics
сnon_trainable_variables
Сlayers
gregularization_losses
 

80
91
:2
;3

80
91
 
▓
 тlayer_regularization_losses
j	variables
ktrainable_variables
Тlayer_metrics
уmetrics
Уnon_trainable_variables
жlayers
lregularization_losses
 
 
 
▓
 Жlayer_regularization_losses
n	variables
otrainable_variables
вlayer_metrics
Вmetrics
ьnon_trainable_variables
Ьlayers
pregularization_losses
 
 
 
▓
 №layer_regularization_losses
r	variables
strainable_variables
­layer_metrics
ыmetrics
Ыnon_trainable_variables
зlayers
tregularization_losses

<0

<0
 
▓
 Зlayer_regularization_losses
v	variables
wtrainable_variables
шlayer_metrics
Шmetrics
эnon_trainable_variables
Эlayers
xregularization_losses
 

=0
>1
?2
@3

=0
>1
 
▓
 щlayer_regularization_losses
{	variables
|trainable_variables
Щlayer_metrics
чmetrics
Чnon_trainable_variables
§layers
}regularization_losses
 
 
 
┤
 ■layer_regularization_losses
	variables
ђtrainable_variables
 layer_metrics
ђmetrics
Ђnon_trainable_variables
ѓlayers
Ђregularization_losses
 
 
 
х
 Ѓlayer_regularization_losses
Ѓ	variables
ёtrainable_variables
ёlayer_metrics
Ёmetrics
єnon_trainable_variables
Єlayers
Ёregularization_losses

A0
B1

A0
B1
 
х
 ѕlayer_regularization_losses
Є	variables
ѕtrainable_variables
Ѕlayer_metrics
іmetrics
Іnon_trainable_variables
їlayers
Ѕregularization_losses
 
 
 
*
50
61
:2
;3
?4
@5
^
	0

1
2
3
4
5
6
7
8
9
10
11
12
 
 
 
х
 Їlayer_regularization_losses
љ	variables
Љtrainable_variables
јlayer_metrics
Јmetrics
љnon_trainable_variables
Љlayers
њregularization_losses

C0
 
 
х
 њlayer_regularization_losses
ћ	variables
Ћtrainable_variables
Њlayer_metrics
ћmetrics
Ћnon_trainable_variables
ќlayers
ќregularization_losses
 

D0
E1
F2
G3
 
 
х
 Ќlayer_regularization_losses
Ў	variables
џtrainable_variables
ўlayer_metrics
Ўmetrics
џnon_trainable_variables
Џlayers
Џregularization_losses
 
 
 
х
 юlayer_regularization_losses
Ю	variables
ъtrainable_variables
Юlayer_metrics
ъmetrics
Ъnon_trainable_variables
аlayers
Ъregularization_losses
 
 
 
х
 Аlayer_regularization_losses
А	variables
бtrainable_variables
бlayer_metrics
Бmetrics
цnon_trainable_variables
Цlayers
Бregularization_losses

H0
 
 
х
 дlayer_regularization_losses
Ц	variables
дtrainable_variables
Дlayer_metrics
еmetrics
Еnon_trainable_variables
фlayers
Дregularization_losses
 

I0
J1
K2
L3
 
 
х
 Фlayer_regularization_losses
ф	variables
Фtrainable_variables
гlayer_metrics
Гmetrics
«non_trainable_variables
»layers
гregularization_losses
 
 
 
х
 ░layer_regularization_losses
«	variables
»trainable_variables
▒layer_metrics
▓metrics
│non_trainable_variables
┤layers
░regularization_losses
 
 
 
х
 хlayer_regularization_losses
▓	variables
│trainable_variables
Хlayer_metrics
иmetrics
Иnon_trainable_variables
╣layers
┤regularization_losses
 
 
 
х
 ║layer_regularization_losses
Х	variables
иtrainable_variables
╗layer_metrics
╝metrics
йnon_trainable_variables
Йlayers
Иregularization_losses

M0
N1
 
 
х
 ┐layer_regularization_losses
║	variables
╗trainable_variables
└layer_metrics
┴metrics
┬non_trainable_variables
├layers
╝regularization_losses
_]
VARIABLE_VALUEYogi/iter_1>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEYogi/beta_1_1@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEYogi/beta_2_1@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEYogi/decay_1?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEYogi/epsilon_1Alayer_with_weights-1/optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE!Yogi/l1_regularization_strength_1Tlayer_with_weights-1/optimizer/l1_regularization_strength/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE!Yogi/l2_regularization_strength_1Tlayer_with_weights-1/optimizer/l2_regularization_strength/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEYogi/learning_rate_1Glayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

─0
V
C0
D1
E2
F3
G4
H5
I6
J7
K8
L9
M10
N11
N
0
1
2
3
4
5
 6
!7
"8
#9
$10
8

┼total

кcount
К	variables
╚	keras_api
 
 
 
 
 
 
 
 

50
61
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

:0
;1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
@1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

C0
 
 
 
 

D0
E1
F2
G3
 
 
 
 
 
 
 
 
 
 
 
 
 
 

H0
 
 
 
 

I0
J1
K2
L3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

M0
N1
 
8

╔total

╩count
╦	variables
╠	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

┼0
к1

К	variables
fd
VARIABLE_VALUEtotal_1Ilayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEcount_1Ilayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

╔0
╩1

╦	variables
ki
VARIABLE_VALUEYogi/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Yogi/batch_normalization/gamma/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEYogi/batch_normalization/beta/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEYogi/conv2d_transpose/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Yogi/batch_normalization_1/gamma/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Yogi/batch_normalization_1/beta/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Yogi/conv2d_transpose_1/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Yogi/batch_normalization_2/gamma/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Yogi/batch_normalization_2/beta/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Yogi/conv2d_transpose_2/kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEYogi/conv2d_transpose_2/bias/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEYogi/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Yogi/batch_normalization/gamma/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEYogi/batch_normalization/beta/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEYogi/conv2d_transpose/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Yogi/batch_normalization_1/gamma/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Yogi/batch_normalization_1/beta/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Yogi/conv2d_transpose_1/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Yogi/batch_normalization_2/gamma/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Yogi/batch_normalization_2/beta/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Yogi/conv2d_transpose_2/kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEYogi/conv2d_transpose_2/bias/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEYogi/conv2d/kernel/vXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Љј
VARIABLE_VALUE"Yogi/batch_normalization_3/gamma/vXvariables/18/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE!Yogi/batch_normalization_3/beta/vXvariables/19/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUEYogi/conv2d_1/kernel/vXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Љј
VARIABLE_VALUE"Yogi/batch_normalization_4/gamma/vXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE!Yogi/batch_normalization_4/beta/vXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUEYogi/dense_1/kernel/vXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEYogi/dense_1/bias/vXvariables/28/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUEYogi/conv2d/kernel/mXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Љј
VARIABLE_VALUE"Yogi/batch_normalization_3/gamma/mXvariables/18/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE!Yogi/batch_normalization_3/beta/mXvariables/19/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUEYogi/conv2d_1/kernel/mXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Љј
VARIABLE_VALUE"Yogi/batch_normalization_4/gamma/mXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
љЇ
VARIABLE_VALUE!Yogi/batch_normalization_4/beta/mXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ёЂ
VARIABLE_VALUEYogi/dense_1/kernel/mXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUEYogi/dense_1/bias/mXvariables/28/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ё
 serving_default_sequential_inputPlaceholder*(
_output_shapes
:         ђ*
dtype0*
shape:         ђ
х	
StatefulPartitionedCallStatefulPartitionedCall serving_default_sequential_inputdense/kernel#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv2d_transpose/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_transpose_1/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_1/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_1/kerneldense_1/bias*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *?
_read_only_resource_inputs!
	
*0
config_proto 

CPU

GPU2*0J 8ѓ */
f*R(
&__inference_signature_wrapper_68632345
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ч"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameYogi/iter/Read/ReadVariableOpYogi/beta_1/Read/ReadVariableOpYogi/beta_2/Read/ReadVariableOpYogi/decay/Read/ReadVariableOp Yogi/epsilon/Read/ReadVariableOp3Yogi/l1_regularization_strength/Read/ReadVariableOp3Yogi/l2_regularization_strength/Read/ReadVariableOp&Yogi/learning_rate/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpYogi/iter_1/Read/ReadVariableOp!Yogi/beta_1_1/Read/ReadVariableOp!Yogi/beta_2_1/Read/ReadVariableOp Yogi/decay_1/Read/ReadVariableOp"Yogi/epsilon_1/Read/ReadVariableOp5Yogi/l1_regularization_strength_1/Read/ReadVariableOp5Yogi/l2_regularization_strength_1/Read/ReadVariableOp(Yogi/learning_rate_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Yogi/dense/kernel/v/Read/ReadVariableOp4Yogi/batch_normalization/gamma/v/Read/ReadVariableOp3Yogi/batch_normalization/beta/v/Read/ReadVariableOp2Yogi/conv2d_transpose/kernel/v/Read/ReadVariableOp6Yogi/batch_normalization_1/gamma/v/Read/ReadVariableOp5Yogi/batch_normalization_1/beta/v/Read/ReadVariableOp4Yogi/conv2d_transpose_1/kernel/v/Read/ReadVariableOp6Yogi/batch_normalization_2/gamma/v/Read/ReadVariableOp5Yogi/batch_normalization_2/beta/v/Read/ReadVariableOp4Yogi/conv2d_transpose_2/kernel/v/Read/ReadVariableOp2Yogi/conv2d_transpose_2/bias/v/Read/ReadVariableOp'Yogi/dense/kernel/m/Read/ReadVariableOp4Yogi/batch_normalization/gamma/m/Read/ReadVariableOp3Yogi/batch_normalization/beta/m/Read/ReadVariableOp2Yogi/conv2d_transpose/kernel/m/Read/ReadVariableOp6Yogi/batch_normalization_1/gamma/m/Read/ReadVariableOp5Yogi/batch_normalization_1/beta/m/Read/ReadVariableOp4Yogi/conv2d_transpose_1/kernel/m/Read/ReadVariableOp6Yogi/batch_normalization_2/gamma/m/Read/ReadVariableOp5Yogi/batch_normalization_2/beta/m/Read/ReadVariableOp4Yogi/conv2d_transpose_2/kernel/m/Read/ReadVariableOp2Yogi/conv2d_transpose_2/bias/m/Read/ReadVariableOp(Yogi/conv2d/kernel/v/Read/ReadVariableOp6Yogi/batch_normalization_3/gamma/v/Read/ReadVariableOp5Yogi/batch_normalization_3/beta/v/Read/ReadVariableOp*Yogi/conv2d_1/kernel/v/Read/ReadVariableOp6Yogi/batch_normalization_4/gamma/v/Read/ReadVariableOp5Yogi/batch_normalization_4/beta/v/Read/ReadVariableOp)Yogi/dense_1/kernel/v/Read/ReadVariableOp'Yogi/dense_1/bias/v/Read/ReadVariableOp(Yogi/conv2d/kernel/m/Read/ReadVariableOp6Yogi/batch_normalization_3/gamma/m/Read/ReadVariableOp5Yogi/batch_normalization_3/beta/m/Read/ReadVariableOp*Yogi/conv2d_1/kernel/m/Read/ReadVariableOp6Yogi/batch_normalization_4/gamma/m/Read/ReadVariableOp5Yogi/batch_normalization_4/beta/m/Read/ReadVariableOp)Yogi/dense_1/kernel/m/Read/ReadVariableOp'Yogi/dense_1/bias/m/Read/ReadVariableOpConst*d
Tin]
[2Y		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__traced_save_68634609
Ф
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Yogi/iterYogi/beta_1Yogi/beta_2
Yogi/decayYogi/epsilonYogi/l1_regularization_strengthYogi/l2_regularization_strengthYogi/learning_ratedense/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_transpose/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_transpose_1/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_1/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancedense_1/kerneldense_1/biasYogi/iter_1Yogi/beta_1_1Yogi/beta_2_1Yogi/decay_1Yogi/epsilon_1!Yogi/l1_regularization_strength_1!Yogi/l2_regularization_strength_1Yogi/learning_rate_1totalcounttotal_1count_1Yogi/dense/kernel/v Yogi/batch_normalization/gamma/vYogi/batch_normalization/beta/vYogi/conv2d_transpose/kernel/v"Yogi/batch_normalization_1/gamma/v!Yogi/batch_normalization_1/beta/v Yogi/conv2d_transpose_1/kernel/v"Yogi/batch_normalization_2/gamma/v!Yogi/batch_normalization_2/beta/v Yogi/conv2d_transpose_2/kernel/vYogi/conv2d_transpose_2/bias/vYogi/dense/kernel/m Yogi/batch_normalization/gamma/mYogi/batch_normalization/beta/mYogi/conv2d_transpose/kernel/m"Yogi/batch_normalization_1/gamma/m!Yogi/batch_normalization_1/beta/m Yogi/conv2d_transpose_1/kernel/m"Yogi/batch_normalization_2/gamma/m!Yogi/batch_normalization_2/beta/m Yogi/conv2d_transpose_2/kernel/mYogi/conv2d_transpose_2/bias/mYogi/conv2d/kernel/v"Yogi/batch_normalization_3/gamma/v!Yogi/batch_normalization_3/beta/vYogi/conv2d_1/kernel/v"Yogi/batch_normalization_4/gamma/v!Yogi/batch_normalization_4/beta/vYogi/dense_1/kernel/vYogi/dense_1/bias/vYogi/conv2d/kernel/m"Yogi/batch_normalization_3/gamma/m!Yogi/batch_normalization_3/beta/mYogi/conv2d_1/kernel/m"Yogi/batch_normalization_4/gamma/m!Yogi/batch_normalization_4/beta/mYogi/dense_1/kernel/mYogi/dense_1/bias/m*c
Tin\
Z2X*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference__traced_restore_68634880жо"
│
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_68630548

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                           @2

IdentityЃ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
в
a
E__inference_reshape_layer_call_and_return_conditional_losses_68633712

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         ђ2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђb:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
┐
a
E__inference_flatten_layer_call_and_return_conditional_losses_68631303

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ12	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ12

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
П
d
E__inference_dropout_layer_call_and_return_conditional_losses_68630462

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/Constј
dropout/MulMulinputsdropout/Const:output:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*B
_output_shapes0
.:,                           ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/y┘
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout/GreaterEqualџ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*B
_output_shapes0
.:,                           ђ2
dropout/CastЋ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout/Mul_1ђ
IdentityIdentitydropout/Mul_1:z:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Є
Ш
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634213

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¤
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ь
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_68634274

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
»Ќ
Е 
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632565

inputs3
/sequential_dense_matmul_readvariableop_resource;
7sequential_batch_normalization_assignmovingavg_68632359=
9sequential_batch_normalization_assignmovingavg_1_68632365H
Dsequential_batch_normalization_batchnorm_mul_readvariableop_resourceD
@sequential_batch_normalization_batchnorm_readvariableop_resourceH
Dsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource<
8sequential_batch_normalization_1_readvariableop_resource>
:sequential_batch_normalization_1_readvariableop_1_resourceM
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceJ
Fsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource<
8sequential_batch_normalization_2_readvariableop_resource>
:sequential_batch_normalization_2_readvariableop_1_resourceM
Isequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceJ
Fsequential_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceA
=sequential_conv2d_transpose_2_biasadd_readvariableop_resource6
2sequential_1_conv2d_conv2d_readvariableop_resource>
:sequential_1_batch_normalization_3_readvariableop_resource@
<sequential_1_batch_normalization_3_readvariableop_1_resourceO
Ksequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource8
4sequential_1_conv2d_1_conv2d_readvariableop_resource>
:sequential_1_batch_normalization_4_readvariableop_resource@
<sequential_1_batch_normalization_4_readvariableop_1_resourceO
Ksequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identityѕбBsequential/batch_normalization/AssignMovingAvg/AssignSubVariableOpб=sequential/batch_normalization/AssignMovingAvg/ReadVariableOpбDsequential/batch_normalization/AssignMovingAvg_1/AssignSubVariableOpб?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOpб7sequential/batch_normalization/batchnorm/ReadVariableOpб;sequential/batch_normalization/batchnorm/mul/ReadVariableOpб/sequential/batch_normalization_1/AssignNewValueб1sequential/batch_normalization_1/AssignNewValue_1б@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбBsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б/sequential/batch_normalization_1/ReadVariableOpб1sequential/batch_normalization_1/ReadVariableOp_1б/sequential/batch_normalization_2/AssignNewValueб1sequential/batch_normalization_2/AssignNewValue_1б@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбBsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б/sequential/batch_normalization_2/ReadVariableOpб1sequential/batch_normalization_2/ReadVariableOp_1б;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpб=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpб4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpб=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpб&sequential/dense/MatMul/ReadVariableOpбBsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбDsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б1sequential_1/batch_normalization_3/ReadVariableOpб3sequential_1/batch_normalization_3/ReadVariableOp_1бBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_1/batch_normalization_4/ReadVariableOpб3sequential_1/batch_normalization_4/ReadVariableOp_1б)sequential_1/conv2d/Conv2D/ReadVariableOpб+sequential_1/conv2d_1/Conv2D/ReadVariableOpб+sequential_1/dense_1/BiasAdd/ReadVariableOpб*sequential_1/dense_1/MatMul/ReadVariableOp┬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђb*
dtype02(
&sequential/dense/MatMul/ReadVariableOpД
sequential/dense/MatMulMatMulinputs.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђb2
sequential/dense/MatMul╚
=sequential/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2?
=sequential/batch_normalization/moments/mean/reduction_indicesѕ
+sequential/batch_normalization/moments/meanMean!sequential/dense/MatMul:product:0Fsequential/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђb*
	keep_dims(2-
+sequential/batch_normalization/moments/mean┌
3sequential/batch_normalization/moments/StopGradientStopGradient4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	ђb25
3sequential/batch_normalization/moments/StopGradientЮ
8sequential/batch_normalization/moments/SquaredDifferenceSquaredDifference!sequential/dense/MatMul:product:0<sequential/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђb2:
8sequential/batch_normalization/moments/SquaredDifferenceл
Asequential/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential/batch_normalization/moments/variance/reduction_indices»
/sequential/batch_normalization/moments/varianceMean<sequential/batch_normalization/moments/SquaredDifference:z:0Jsequential/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђb*
	keep_dims(21
/sequential/batch_normalization/moments/varianceя
.sequential/batch_normalization/moments/SqueezeSqueeze4sequential/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:ђb*
squeeze_dims
 20
.sequential/batch_normalization/moments/SqueezeТ
0sequential/batch_normalization/moments/Squeeze_1Squeeze8sequential/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:ђb*
squeeze_dims
 22
0sequential/batch_normalization/moments/Squeeze_1Ф
4sequential/batch_normalization/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@sequential/batch_normalization/AssignMovingAvg/68632359*
_output_shapes
: *
dtype0*
valueB
 *
О#<26
4sequential/batch_normalization/AssignMovingAvg/decayз
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp7sequential_batch_normalization_assignmovingavg_68632359*
_output_shapes	
:ђb*
dtype02?
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOpЈ
2sequential/batch_normalization/AssignMovingAvg/subSubEsequential/batch_normalization/AssignMovingAvg/ReadVariableOp:value:07sequential/batch_normalization/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*J
_class@
><loc:@sequential/batch_normalization/AssignMovingAvg/68632359*
_output_shapes	
:ђb24
2sequential/batch_normalization/AssignMovingAvg/subє
2sequential/batch_normalization/AssignMovingAvg/mulMul6sequential/batch_normalization/AssignMovingAvg/sub:z:0=sequential/batch_normalization/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*J
_class@
><loc:@sequential/batch_normalization/AssignMovingAvg/68632359*
_output_shapes	
:ђb24
2sequential/batch_normalization/AssignMovingAvg/mulь
Bsequential/batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp7sequential_batch_normalization_assignmovingavg_686323596sequential/batch_normalization/AssignMovingAvg/mul:z:0>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*J
_class@
><loc:@sequential/batch_normalization/AssignMovingAvg/68632359*
_output_shapes
 *
dtype02D
Bsequential/batch_normalization/AssignMovingAvg/AssignSubVariableOp▒
6sequential/batch_normalization/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*L
_classB
@>loc:@sequential/batch_normalization/AssignMovingAvg_1/68632365*
_output_shapes
: *
dtype0*
valueB
 *
О#<28
6sequential/batch_normalization/AssignMovingAvg_1/decayщ
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp9sequential_batch_normalization_assignmovingavg_1_68632365*
_output_shapes	
:ђb*
dtype02A
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOpЎ
4sequential/batch_normalization/AssignMovingAvg_1/subSubGsequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:09sequential/batch_normalization/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@sequential/batch_normalization/AssignMovingAvg_1/68632365*
_output_shapes	
:ђb26
4sequential/batch_normalization/AssignMovingAvg_1/subљ
4sequential/batch_normalization/AssignMovingAvg_1/mulMul8sequential/batch_normalization/AssignMovingAvg_1/sub:z:0?sequential/batch_normalization/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@sequential/batch_normalization/AssignMovingAvg_1/68632365*
_output_shapes	
:ђb26
4sequential/batch_normalization/AssignMovingAvg_1/mulщ
Dsequential/batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp9sequential_batch_normalization_assignmovingavg_1_686323658sequential/batch_normalization/AssignMovingAvg_1/mul:z:0@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*L
_classB
@>loc:@sequential/batch_normalization/AssignMovingAvg_1/68632365*
_output_shapes
 *
dtype02F
Dsequential/batch_normalization/AssignMovingAvg_1/AssignSubVariableOpЦ
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:20
.sequential/batch_normalization/batchnorm/add/y 
,sequential/batch_normalization/batchnorm/addAddV29sequential/batch_normalization/moments/Squeeze_1:output:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђb2.
,sequential/batch_normalization/batchnorm/add┴
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ђb20
.sequential/batch_normalization/batchnorm/RsqrtЧ
;sequential/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpDsequential_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђb*
dtype02=
;sequential/batch_normalization/batchnorm/mul/ReadVariableOpѓ
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0Csequential/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђb2.
,sequential/batch_normalization/batchnorm/mul№
.sequential/batch_normalization/batchnorm/mul_1Mul!sequential/dense/MatMul:product:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђb20
.sequential/batch_normalization/batchnorm/mul_1Э
.sequential/batch_normalization/batchnorm/mul_2Mul7sequential/batch_normalization/moments/Squeeze:output:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђb20
.sequential/batch_normalization/batchnorm/mul_2­
7sequential/batch_normalization/batchnorm/ReadVariableOpReadVariableOp@sequential_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:ђb*
dtype029
7sequential/batch_normalization/batchnorm/ReadVariableOp■
,sequential/batch_normalization/batchnorm/subSub?sequential/batch_normalization/batchnorm/ReadVariableOp:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђb2.
,sequential/batch_normalization/batchnorm/subѓ
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђb20
.sequential/batch_normalization/batchnorm/add_1┐
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu2sequential/batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         ђb*
alpha%џЎЎ>2"
 sequential/leaky_re_lu/LeakyReluњ
sequential/reshape/ShapeShape.sequential/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
sequential/reshape/Shapeџ
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/reshape/strided_slice/stackъ
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_1ъ
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_2н
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/reshape/strided_sliceі
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/1і
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/2І
"sequential/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2$
"sequential/reshape/Reshape/shape/3г
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0+sequential/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 sequential/reshape/Reshape/shape┘
sequential/reshape/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0)sequential/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђ2
sequential/reshape/ReshapeЎ
!sequential/conv2d_transpose/ShapeShape#sequential/reshape/Reshape:output:0*
T0*
_output_shapes
:2#
!sequential/conv2d_transpose/Shapeг
/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential/conv2d_transpose/strided_slice/stack░
1sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice/stack_1░
1sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice/stack_2і
)sequential/conv2d_transpose/strided_sliceStridedSlice*sequential/conv2d_transpose/Shape:output:08sequential/conv2d_transpose/strided_slice/stack:output:0:sequential/conv2d_transpose/strided_slice/stack_1:output:0:sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential/conv2d_transpose/strided_sliceї
#sequential/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/conv2d_transpose/stack/1ї
#sequential/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/conv2d_transpose/stack/2Ї
#sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2%
#sequential/conv2d_transpose/stack/3║
!sequential/conv2d_transpose/stackPack2sequential/conv2d_transpose/strided_slice:output:0,sequential/conv2d_transpose/stack/1:output:0,sequential/conv2d_transpose/stack/2:output:0,sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!sequential/conv2d_transpose/stack░
1sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose/strided_slice_1/stack┤
3sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_1/stack_1┤
3sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_1/stack_2ћ
+sequential/conv2d_transpose/strided_slice_1StridedSlice*sequential/conv2d_transpose/stack:output:0:sequential/conv2d_transpose/strided_slice_1/stack:output:0<sequential/conv2d_transpose/strided_slice_1/stack_1:output:0<sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose/strided_slice_1Ѕ
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02=
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpВ
,sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput*sequential/conv2d_transpose/stack:output:0Csequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0#sequential/reshape/Reshape:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2.
,sequential/conv2d_transpose/conv2d_transposeп
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/sequential/batch_normalization_1/ReadVariableOpя
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1І
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЉ
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1м
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV35sequential/conv2d_transpose/conv2d_transpose:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<23
1sequential/batch_normalization_1/FusedBatchNormV3з
/sequential/batch_normalization_1/AssignNewValueAssignVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource>sequential/batch_normalization_1/FusedBatchNormV3:batch_mean:0A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*\
_classR
PNloc:@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype021
/sequential/batch_normalization_1/AssignNewValueЂ
1sequential/batch_normalization_1/AssignNewValue_1AssignVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceBsequential/batch_normalization_1/FusedBatchNormV3:batch_variance:0C^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*^
_classT
RPloc:@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype023
1sequential/batch_normalization_1/AssignNewValue_1╬
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu5sequential/batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>2$
"sequential/leaky_re_lu_1/LeakyReluЅ
 sequential/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2"
 sequential/dropout/dropout/Const▀
sequential/dropout/dropout/MulMul0sequential/leaky_re_lu_1/LeakyRelu:activations:0)sequential/dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2 
sequential/dropout/dropout/Mulц
 sequential/dropout/dropout/ShapeShape0sequential/leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2"
 sequential/dropout/dropout/ShapeШ
7sequential/dropout/dropout/random_uniform/RandomUniformRandomUniform)sequential/dropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype029
7sequential/dropout/dropout/random_uniform/RandomUniformЏ
)sequential/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2+
)sequential/dropout/dropout/GreaterEqual/yЊ
'sequential/dropout/dropout/GreaterEqualGreaterEqual@sequential/dropout/dropout/random_uniform/RandomUniform:output:02sequential/dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2)
'sequential/dropout/dropout/GreaterEqual┴
sequential/dropout/dropout/CastCast+sequential/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2!
sequential/dropout/dropout/Cast¤
 sequential/dropout/dropout/Mul_1Mul"sequential/dropout/dropout/Mul:z:0#sequential/dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2"
 sequential/dropout/dropout/Mul_1ъ
#sequential/conv2d_transpose_1/ShapeShape$sequential/dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_1/Shape░
1sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose_1/strided_slice/stack┤
3sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice/stack_1┤
3sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice/stack_2ќ
+sequential/conv2d_transpose_1/strided_sliceStridedSlice,sequential/conv2d_transpose_1/Shape:output:0:sequential/conv2d_transpose_1/strided_slice/stack:output:0<sequential/conv2d_transpose_1/strided_slice/stack_1:output:0<sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose_1/strided_sliceљ
%sequential/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_1/stack/1љ
%sequential/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_1/stack/2љ
%sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential/conv2d_transpose_1/stack/3к
#sequential/conv2d_transpose_1/stackPack4sequential/conv2d_transpose_1/strided_slice:output:0.sequential/conv2d_transpose_1/stack/1:output:0.sequential/conv2d_transpose_1/stack/2:output:0.sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_1/stack┤
3sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/conv2d_transpose_1/strided_slice_1/stackИ
5sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_1/stack_1И
5sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_1/stack_2а
-sequential/conv2d_transpose_1/strided_slice_1StridedSlice,sequential/conv2d_transpose_1/stack:output:0<sequential/conv2d_transpose_1/strided_slice_1/stack:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_1/strided_slice_1ј
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02?
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpЗ
.sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_1/stack:output:0Esequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0$sequential/dropout/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
20
.sequential/conv2d_transpose_1/conv2d_transposeО
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential/batch_normalization_2/ReadVariableOpП
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1sequential/batch_normalization_2/ReadVariableOp_1і
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpљ
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1¤
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV37sequential/conv2d_transpose_1/conv2d_transpose:output:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<23
1sequential/batch_normalization_2/FusedBatchNormV3з
/sequential/batch_normalization_2/AssignNewValueAssignVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource>sequential/batch_normalization_2/FusedBatchNormV3:batch_mean:0A^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*\
_classR
PNloc:@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype021
/sequential/batch_normalization_2/AssignNewValueЂ
1sequential/batch_normalization_2/AssignNewValue_1AssignVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceBsequential/batch_normalization_2/FusedBatchNormV3:batch_variance:0C^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*^
_classT
RPloc:@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype023
1sequential/batch_normalization_2/AssignNewValue_1═
"sequential/leaky_re_lu_2/LeakyRelu	LeakyRelu5sequential/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>2$
"sequential/leaky_re_lu_2/LeakyReluЇ
"sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2$
"sequential/dropout_1/dropout/ConstС
 sequential/dropout_1/dropout/MulMul0sequential/leaky_re_lu_2/LeakyRelu:activations:0+sequential/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:         @2"
 sequential/dropout_1/dropout/Mulе
"sequential/dropout_1/dropout/ShapeShape0sequential/leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:2$
"sequential/dropout_1/dropout/Shapeч
9sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02;
9sequential/dropout_1/dropout/random_uniform/RandomUniformЪ
+sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2-
+sequential/dropout_1/dropout/GreaterEqual/yџ
)sequential/dropout_1/dropout/GreaterEqualGreaterEqualBsequential/dropout_1/dropout/random_uniform/RandomUniform:output:04sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2+
)sequential/dropout_1/dropout/GreaterEqualк
!sequential/dropout_1/dropout/CastCast-sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2#
!sequential/dropout_1/dropout/Castо
"sequential/dropout_1/dropout/Mul_1Mul$sequential/dropout_1/dropout/Mul:z:0%sequential/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2$
"sequential/dropout_1/dropout/Mul_1а
#sequential/conv2d_transpose_2/ShapeShape&sequential/dropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_2/Shape░
1sequential/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose_2/strided_slice/stack┤
3sequential/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_2/strided_slice/stack_1┤
3sequential/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_2/strided_slice/stack_2ќ
+sequential/conv2d_transpose_2/strided_sliceStridedSlice,sequential/conv2d_transpose_2/Shape:output:0:sequential/conv2d_transpose_2/strided_slice/stack:output:0<sequential/conv2d_transpose_2/strided_slice/stack_1:output:0<sequential/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose_2/strided_sliceљ
%sequential/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_2/stack/1љ
%sequential/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_2/stack/2љ
%sequential/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_2/stack/3к
#sequential/conv2d_transpose_2/stackPack4sequential/conv2d_transpose_2/strided_slice:output:0.sequential/conv2d_transpose_2/stack/1:output:0.sequential/conv2d_transpose_2/stack/2:output:0.sequential/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_2/stack┤
3sequential/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/conv2d_transpose_2/strided_slice_1/stackИ
5sequential/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_2/strided_slice_1/stack_1И
5sequential/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_2/strided_slice_1/stack_2а
-sequential/conv2d_transpose_2/strided_slice_1StridedSlice,sequential/conv2d_transpose_2/stack:output:0<sequential/conv2d_transpose_2/strided_slice_1/stack:output:0>sequential/conv2d_transpose_2/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_2/strided_slice_1Ї
=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02?
=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpШ
.sequential/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_2/stack:output:0Esequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0&sequential/dropout_1/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
20
.sequential/conv2d_transpose_2/conv2d_transposeТ
4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp=sequential_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpі
%sequential/conv2d_transpose_2/BiasAddBiasAdd7sequential/conv2d_transpose_2/conv2d_transpose:output:0<sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2'
%sequential/conv2d_transpose_2/BiasAdd║
"sequential/conv2d_transpose_2/TanhTanh.sequential/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:         2$
"sequential/conv2d_transpose_2/Tanhю
!sequential_1/gaussian_noise/ShapeShape&sequential/conv2d_transpose_2/Tanh:y:0*
T0*
_output_shapes
:2#
!sequential_1/gaussian_noise/ShapeЦ
.sequential_1/gaussian_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.sequential_1/gaussian_noise/random_normal/meanЕ
0sequential_1/gaussian_noise/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>22
0sequential_1/gaussian_noise/random_normal/stddevф
>sequential_1/gaussian_noise/random_normal/RandomStandardNormalRandomStandardNormal*sequential_1/gaussian_noise/Shape:output:0*
T0*/
_output_shapes
:         *
dtype0*
seed▒ т)*
seed2Ћ§«2@
>sequential_1/gaussian_noise/random_normal/RandomStandardNormalБ
-sequential_1/gaussian_noise/random_normal/mulMulGsequential_1/gaussian_noise/random_normal/RandomStandardNormal:output:09sequential_1/gaussian_noise/random_normal/stddev:output:0*
T0*/
_output_shapes
:         2/
-sequential_1/gaussian_noise/random_normal/mulЃ
)sequential_1/gaussian_noise/random_normalAdd1sequential_1/gaussian_noise/random_normal/mul:z:07sequential_1/gaussian_noise/random_normal/mean:output:0*
T0*/
_output_shapes
:         2+
)sequential_1/gaussian_noise/random_normal▄
sequential_1/gaussian_noise/addAddV2&sequential/conv2d_transpose_2/Tanh:y:0-sequential_1/gaussian_noise/random_normal:z:0*
T0*/
_output_shapes
:         2!
sequential_1/gaussian_noise/addЛ
)sequential_1/conv2d/Conv2D/ReadVariableOpReadVariableOp2sequential_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02+
)sequential_1/conv2d/Conv2D/ReadVariableOpЧ
sequential_1/conv2d/Conv2DConv2D#sequential_1/gaussian_noise/add:z:01sequential_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
sequential_1/conv2d/Conv2DП
1sequential_1/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_1/batch_normalization_3/ReadVariableOpс
3sequential_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3sequential_1/batch_normalization_3/ReadVariableOp_1љ
Bsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpќ
Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1╣
3sequential_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3#sequential_1/conv2d/Conv2D:output:09sequential_1/batch_normalization_3/ReadVariableOp:value:0;sequential_1/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 25
3sequential_1/batch_normalization_3/FusedBatchNormV3М
$sequential_1/leaky_re_lu_3/LeakyRelu	LeakyRelu7sequential_1/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>2&
$sequential_1/leaky_re_lu_3/LeakyReluЉ
$sequential_1/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2&
$sequential_1/dropout_2/dropout/ConstВ
"sequential_1/dropout_2/dropout/MulMul2sequential_1/leaky_re_lu_3/LeakyRelu:activations:0-sequential_1/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:         @2$
"sequential_1/dropout_2/dropout/Mul«
$sequential_1/dropout_2/dropout/ShapeShape2sequential_1/leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2&
$sequential_1/dropout_2/dropout/ShapeЂ
;sequential_1/dropout_2/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02=
;sequential_1/dropout_2/dropout/random_uniform/RandomUniformБ
-sequential_1/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2/
-sequential_1/dropout_2/dropout/GreaterEqual/yб
+sequential_1/dropout_2/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_2/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2-
+sequential_1/dropout_2/dropout/GreaterEqual╠
#sequential_1/dropout_2/dropout/CastCast/sequential_1/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2%
#sequential_1/dropout_2/dropout/Castя
$sequential_1/dropout_2/dropout/Mul_1Mul&sequential_1/dropout_2/dropout/Mul:z:0'sequential_1/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2&
$sequential_1/dropout_2/dropout/Mul_1п
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02-
+sequential_1/conv2d_1/Conv2D/ReadVariableOpѕ
sequential_1/conv2d_1/Conv2DConv2D(sequential_1/dropout_2/dropout/Mul_1:z:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_1/conv2d_1/Conv2Dя
1sequential_1/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1sequential_1/batch_normalization_4/ReadVariableOpС
3sequential_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype025
3sequential_1/batch_normalization_4/ReadVariableOp_1Љ
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpЌ
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02F
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1└
3sequential_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%sequential_1/conv2d_1/Conv2D:output:09sequential_1/batch_normalization_4/ReadVariableOp:value:0;sequential_1/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 25
3sequential_1/batch_normalization_4/FusedBatchNormV3Љ
$sequential_1/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2&
$sequential_1/dropout_3/dropout/ConstЫ
"sequential_1/dropout_3/dropout/MulMul7sequential_1/batch_normalization_4/FusedBatchNormV3:y:0-sequential_1/dropout_3/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2$
"sequential_1/dropout_3/dropout/Mul│
$sequential_1/dropout_3/dropout/ShapeShape7sequential_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2&
$sequential_1/dropout_3/dropout/Shapeѓ
;sequential_1/dropout_3/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_3/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02=
;sequential_1/dropout_3/dropout/random_uniform/RandomUniformБ
-sequential_1/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2/
-sequential_1/dropout_3/dropout/GreaterEqual/yБ
+sequential_1/dropout_3/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_3/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_3/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2-
+sequential_1/dropout_3/dropout/GreaterEqual═
#sequential_1/dropout_3/dropout/CastCast/sequential_1/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2%
#sequential_1/dropout_3/dropout/Cast▀
$sequential_1/dropout_3/dropout/Mul_1Mul&sequential_1/dropout_3/dropout/Mul:z:0'sequential_1/dropout_3/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2&
$sequential_1/dropout_3/dropout/Mul_1┼
$sequential_1/leaky_re_lu_4/LeakyRelu	LeakyRelu(sequential_1/dropout_3/dropout/Mul_1:z:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>2&
$sequential_1/leaky_re_lu_4/LeakyReluЅ
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
sequential_1/flatten/ConstМ
sequential_1/flatten/ReshapeReshape2sequential_1/leaky_re_lu_4/LeakyRelu:activations:0#sequential_1/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ12
sequential_1/flatten/Reshape═
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOpЛ
sequential_1/dense_1/MatMulMatMul%sequential_1/flatten/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_1/MatMul╦
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpН
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_1/BiasAddа
sequential_1/dense_1/SigmoidSigmoid%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential_1/dense_1/Sigmoidр
IdentityIdentity sequential_1/dense_1/Sigmoid:y:0C^sequential/batch_normalization/AssignMovingAvg/AssignSubVariableOp>^sequential/batch_normalization/AssignMovingAvg/ReadVariableOpE^sequential/batch_normalization/AssignMovingAvg_1/AssignSubVariableOp@^sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp8^sequential/batch_normalization/batchnorm/ReadVariableOp<^sequential/batch_normalization/batchnorm/mul/ReadVariableOp0^sequential/batch_normalization_1/AssignNewValue2^sequential/batch_normalization_1/AssignNewValue_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_10^sequential/batch_normalization_2/AssignNewValue2^sequential/batch_normalization_2/AssignNewValue_1A^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_2/ReadVariableOp2^sequential/batch_normalization_2/ReadVariableOp_1<^sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp>^sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp5^sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp>^sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOpC^sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_3/ReadVariableOp4^sequential_1/batch_normalization_3/ReadVariableOp_1C^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_4/ReadVariableOp4^sequential_1/batch_normalization_4/ReadVariableOp_1*^sequential_1/conv2d/Conv2D/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::2ѕ
Bsequential/batch_normalization/AssignMovingAvg/AssignSubVariableOpBsequential/batch_normalization/AssignMovingAvg/AssignSubVariableOp2~
=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp=sequential/batch_normalization/AssignMovingAvg/ReadVariableOp2ї
Dsequential/batch_normalization/AssignMovingAvg_1/AssignSubVariableOpDsequential/batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2ѓ
?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp?sequential/batch_normalization/AssignMovingAvg_1/ReadVariableOp2r
7sequential/batch_normalization/batchnorm/ReadVariableOp7sequential/batch_normalization/batchnorm/ReadVariableOp2z
;sequential/batch_normalization/batchnorm/mul/ReadVariableOp;sequential/batch_normalization/batchnorm/mul/ReadVariableOp2b
/sequential/batch_normalization_1/AssignNewValue/sequential/batch_normalization_1/AssignNewValue2f
1sequential/batch_normalization_1/AssignNewValue_11sequential/batch_normalization_1/AssignNewValue_12ё
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2ѕ
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12b
/sequential/batch_normalization_2/AssignNewValue/sequential/batch_normalization_2/AssignNewValue2f
1sequential/batch_normalization_2/AssignNewValue_11sequential/batch_normalization_2/AssignNewValue_12ё
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2ѕ
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_2/ReadVariableOp/sequential/batch_normalization_2/ReadVariableOp2f
1sequential/batch_normalization_2/ReadVariableOp_11sequential/batch_normalization_2/ReadVariableOp_12z
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2~
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2l
4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp2~
=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2ѕ
Bsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_3/ReadVariableOp1sequential_1/batch_normalization_3/ReadVariableOp2j
3sequential_1/batch_normalization_3/ReadVariableOp_13sequential_1/batch_normalization_3/ReadVariableOp_12ѕ
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_4/ReadVariableOp1sequential_1/batch_normalization_4/ReadVariableOp2j
3sequential_1/batch_normalization_4/ReadVariableOp_13sequential_1/batch_normalization_4/ReadVariableOp_12V
)sequential_1/conv2d/Conv2D/ReadVariableOp)sequential_1/conv2d/Conv2D/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
├
Ш
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68630857

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ж
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_68634109

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
┼
f
G__inference_dropout_2_layer_call_and_return_conditional_losses_68631146

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ф
Ф
8__inference_batch_normalization_4_layer_call_fn_68634195

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall╗
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_686309882
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
д
г
J__inference_sequential_2_layer_call_and_return_conditional_losses_68631947
sequential_input
sequential_68631597
sequential_68631599
sequential_68631601
sequential_68631603
sequential_68631605
sequential_68631607
sequential_68631609
sequential_68631611
sequential_68631613
sequential_68631615
sequential_68631617
sequential_68631619
sequential_68631621
sequential_68631623
sequential_68631625
sequential_68631627
sequential_68631629
sequential_1_68631921
sequential_1_68631923
sequential_1_68631925
sequential_1_68631927
sequential_1_68631929
sequential_1_68631931
sequential_1_68631933
sequential_1_68631935
sequential_1_68631937
sequential_1_68631939
sequential_1_68631941
sequential_1_68631943
identityѕб"sequential/StatefulPartitionedCallб$sequential_1/StatefulPartitionedCallъ
"sequential/StatefulPartitionedCallStatefulPartitionedCallsequential_inputsequential_68631597sequential_68631599sequential_68631601sequential_68631603sequential_68631605sequential_68631607sequential_68631609sequential_68631611sequential_68631613sequential_68631615sequential_68631617sequential_68631619sequential_68631621sequential_68631623sequential_68631625sequential_68631627sequential_68631629*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *-
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_686306722$
"sequential/StatefulPartitionedCallл
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_68631921sequential_1_68631923sequential_1_68631925sequential_1_68631927sequential_1_68631929sequential_1_68631931sequential_1_68631933sequential_1_68631935sequential_1_68631937sequential_1_68631939sequential_1_68631941sequential_1_68631943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686318052&
$sequential_1/StatefulPartitionedCall═
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Z V
(
_output_shapes
:         ђ
*
_user_specified_namesequential_input
К
e
,__inference_dropout_2_layer_call_fn_68634114

inputs
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_686311462
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╦
e
,__inference_dropout_3_layer_call_fn_68634279

inputs
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_686312662
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
ф
Ф
8__inference_batch_normalization_1_layer_call_fn_68633781

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall╗
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_686301072
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
┴
{
5__inference_conv2d_transpose_1_layer_call_fn_68630157

inputs
unknown
identityѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_686301492
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           ђ:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
же
░(
!__inference__traced_save_68634609
file_prefix(
$savev2_yogi_iter_read_readvariableop	*
&savev2_yogi_beta_1_read_readvariableop*
&savev2_yogi_beta_2_read_readvariableop)
%savev2_yogi_decay_read_readvariableop+
'savev2_yogi_epsilon_read_readvariableop>
:savev2_yogi_l1_regularization_strength_read_readvariableop>
:savev2_yogi_l2_regularization_strength_read_readvariableop1
-savev2_yogi_learning_rate_read_readvariableop+
'savev2_dense_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop*
&savev2_yogi_iter_1_read_readvariableop	,
(savev2_yogi_beta_1_1_read_readvariableop,
(savev2_yogi_beta_2_1_read_readvariableop+
'savev2_yogi_decay_1_read_readvariableop-
)savev2_yogi_epsilon_1_read_readvariableop@
<savev2_yogi_l1_regularization_strength_1_read_readvariableop@
<savev2_yogi_l2_regularization_strength_1_read_readvariableop3
/savev2_yogi_learning_rate_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_yogi_dense_kernel_v_read_readvariableop?
;savev2_yogi_batch_normalization_gamma_v_read_readvariableop>
:savev2_yogi_batch_normalization_beta_v_read_readvariableop=
9savev2_yogi_conv2d_transpose_kernel_v_read_readvariableopA
=savev2_yogi_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_yogi_batch_normalization_1_beta_v_read_readvariableop?
;savev2_yogi_conv2d_transpose_1_kernel_v_read_readvariableopA
=savev2_yogi_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_yogi_batch_normalization_2_beta_v_read_readvariableop?
;savev2_yogi_conv2d_transpose_2_kernel_v_read_readvariableop=
9savev2_yogi_conv2d_transpose_2_bias_v_read_readvariableop2
.savev2_yogi_dense_kernel_m_read_readvariableop?
;savev2_yogi_batch_normalization_gamma_m_read_readvariableop>
:savev2_yogi_batch_normalization_beta_m_read_readvariableop=
9savev2_yogi_conv2d_transpose_kernel_m_read_readvariableopA
=savev2_yogi_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_yogi_batch_normalization_1_beta_m_read_readvariableop?
;savev2_yogi_conv2d_transpose_1_kernel_m_read_readvariableopA
=savev2_yogi_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_yogi_batch_normalization_2_beta_m_read_readvariableop?
;savev2_yogi_conv2d_transpose_2_kernel_m_read_readvariableop=
9savev2_yogi_conv2d_transpose_2_bias_m_read_readvariableop3
/savev2_yogi_conv2d_kernel_v_read_readvariableopA
=savev2_yogi_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_yogi_batch_normalization_3_beta_v_read_readvariableop5
1savev2_yogi_conv2d_1_kernel_v_read_readvariableopA
=savev2_yogi_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_yogi_batch_normalization_4_beta_v_read_readvariableop4
0savev2_yogi_dense_1_kernel_v_read_readvariableop2
.savev2_yogi_dense_1_bias_v_read_readvariableop3
/savev2_yogi_conv2d_kernel_m_read_readvariableopA
=savev2_yogi_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_yogi_batch_normalization_3_beta_m_read_readvariableop5
1savev2_yogi_conv2d_1_kernel_m_read_readvariableopA
=savev2_yogi_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_yogi_batch_normalization_4_beta_m_read_readvariableop4
0savev2_yogi_dense_1_kernel_m_read_readvariableop2
.savev2_yogi_dense_1_bias_m_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameњ+
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*ц*
valueџ*BЌ*XB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUEB?optimizer/l1_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEB?optimizer/l2_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/optimizer/l1_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/optimizer/l2_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/18/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/19/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/28/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/18/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/19/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/28/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╗
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*┼
value╗BИXB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesэ&
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_yogi_iter_read_readvariableop&savev2_yogi_beta_1_read_readvariableop&savev2_yogi_beta_2_read_readvariableop%savev2_yogi_decay_read_readvariableop'savev2_yogi_epsilon_read_readvariableop:savev2_yogi_l1_regularization_strength_read_readvariableop:savev2_yogi_l2_regularization_strength_read_readvariableop-savev2_yogi_learning_rate_read_readvariableop'savev2_dense_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop&savev2_yogi_iter_1_read_readvariableop(savev2_yogi_beta_1_1_read_readvariableop(savev2_yogi_beta_2_1_read_readvariableop'savev2_yogi_decay_1_read_readvariableop)savev2_yogi_epsilon_1_read_readvariableop<savev2_yogi_l1_regularization_strength_1_read_readvariableop<savev2_yogi_l2_regularization_strength_1_read_readvariableop/savev2_yogi_learning_rate_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_yogi_dense_kernel_v_read_readvariableop;savev2_yogi_batch_normalization_gamma_v_read_readvariableop:savev2_yogi_batch_normalization_beta_v_read_readvariableop9savev2_yogi_conv2d_transpose_kernel_v_read_readvariableop=savev2_yogi_batch_normalization_1_gamma_v_read_readvariableop<savev2_yogi_batch_normalization_1_beta_v_read_readvariableop;savev2_yogi_conv2d_transpose_1_kernel_v_read_readvariableop=savev2_yogi_batch_normalization_2_gamma_v_read_readvariableop<savev2_yogi_batch_normalization_2_beta_v_read_readvariableop;savev2_yogi_conv2d_transpose_2_kernel_v_read_readvariableop9savev2_yogi_conv2d_transpose_2_bias_v_read_readvariableop.savev2_yogi_dense_kernel_m_read_readvariableop;savev2_yogi_batch_normalization_gamma_m_read_readvariableop:savev2_yogi_batch_normalization_beta_m_read_readvariableop9savev2_yogi_conv2d_transpose_kernel_m_read_readvariableop=savev2_yogi_batch_normalization_1_gamma_m_read_readvariableop<savev2_yogi_batch_normalization_1_beta_m_read_readvariableop;savev2_yogi_conv2d_transpose_1_kernel_m_read_readvariableop=savev2_yogi_batch_normalization_2_gamma_m_read_readvariableop<savev2_yogi_batch_normalization_2_beta_m_read_readvariableop;savev2_yogi_conv2d_transpose_2_kernel_m_read_readvariableop9savev2_yogi_conv2d_transpose_2_bias_m_read_readvariableop/savev2_yogi_conv2d_kernel_v_read_readvariableop=savev2_yogi_batch_normalization_3_gamma_v_read_readvariableop<savev2_yogi_batch_normalization_3_beta_v_read_readvariableop1savev2_yogi_conv2d_1_kernel_v_read_readvariableop=savev2_yogi_batch_normalization_4_gamma_v_read_readvariableop<savev2_yogi_batch_normalization_4_beta_v_read_readvariableop0savev2_yogi_dense_1_kernel_v_read_readvariableop.savev2_yogi_dense_1_bias_v_read_readvariableop/savev2_yogi_conv2d_kernel_m_read_readvariableop=savev2_yogi_batch_normalization_3_gamma_m_read_readvariableop<savev2_yogi_batch_normalization_3_beta_m_read_readvariableop1savev2_yogi_conv2d_1_kernel_m_read_readvariableop=savev2_yogi_batch_normalization_4_gamma_m_read_readvariableop<savev2_yogi_batch_normalization_4_beta_m_read_readvariableop0savev2_yogi_dense_1_kernel_m_read_readvariableop.savev2_yogi_dense_1_bias_m_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *f
dtypes\
Z2X		2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╠
_input_shapes║
и: : : : : : : : : :
ђђb:ђb:ђb:ђb:ђb:ђђ:ђ:ђ:ђ:ђ:@ђ:@:@:@:@:@::@:@:@:@:@:@ђ:ђ:ђ:ђ:ђ:	ђ1:: : : : : : : : : : : : :
ђђb:ђb:ђb:ђђ:ђ:ђ:@ђ:@:@:@::
ђђb:ђb:ђb:ђђ:ђ:ђ:@ђ:@:@:@::@:@:@:@ђ:ђ:ђ:	ђ1::@:@:@:@ђ:ђ:ђ:	ђ1:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&	"
 
_output_shapes
:
ђђb:!


_output_shapes	
:ђb:!

_output_shapes	
:ђb:!

_output_shapes	
:ђb:!

_output_shapes	
:ђb:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:-)
'
_output_shapes
:@ђ: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@ђ:! 

_output_shapes	
:ђ:!!

_output_shapes	
:ђ:!"

_output_shapes	
:ђ:!#

_output_shapes	
:ђ:%$!

_output_shapes
:	ђ1: %

_output_shapes
::&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :&2"
 
_output_shapes
:
ђђb:!3

_output_shapes	
:ђb:!4

_output_shapes	
:ђb:.5*
(
_output_shapes
:ђђ:!6

_output_shapes	
:ђ:!7

_output_shapes	
:ђ:-8)
'
_output_shapes
:@ђ: 9

_output_shapes
:@: :

_output_shapes
:@:,;(
&
_output_shapes
:@: <

_output_shapes
::&="
 
_output_shapes
:
ђђb:!>

_output_shapes	
:ђb:!?

_output_shapes	
:ђb:.@*
(
_output_shapes
:ђђ:!A

_output_shapes	
:ђ:!B

_output_shapes	
:ђ:-C)
'
_output_shapes
:@ђ: D

_output_shapes
:@: E

_output_shapes
:@:,F(
&
_output_shapes
:@: G

_output_shapes
::,H(
&
_output_shapes
:@: I

_output_shapes
:@: J

_output_shapes
:@:-K)
'
_output_shapes
:@ђ:!L

_output_shapes	
:ђ:!M

_output_shapes	
:ђ:%N!

_output_shapes
:	ђ1: O

_output_shapes
::,P(
&
_output_shapes
:@: Q

_output_shapes
:@: R

_output_shapes
:@:-S)
'
_output_shapes
:@ђ:!T

_output_shapes	
:ђ:!U

_output_shapes	
:ђ:%V!

_output_shapes
:	ђ1: W

_output_shapes
::X

_output_shapes
: 
├
Ш
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68633976

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╝
Е
6__inference_batch_normalization_layer_call_fn_68633675

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_686299312
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђb::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
═
f
G__inference_dropout_3_layer_call_and_return_conditional_losses_68634269

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/yК
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
░

k
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_68631014

inputs
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
random_normal/stddevН
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:         *
dtype0*
seed▒ т)*
seed2Ек/2$
"random_normal/RandomStandardNormal│
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:         2
random_normal/mulЊ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:         2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:         2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┴
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_68630442

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,                           ђ*
alpha%џЎЎ>2
	LeakyReluє
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
╗
H
,__inference_dropout_2_layer_call_fn_68634119

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_686311512
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ѕ
б
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632078

inputs
sequential_68632017
sequential_68632019
sequential_68632021
sequential_68632023
sequential_68632025
sequential_68632027
sequential_68632029
sequential_68632031
sequential_68632033
sequential_68632035
sequential_68632037
sequential_68632039
sequential_68632041
sequential_68632043
sequential_68632045
sequential_68632047
sequential_68632049
sequential_1_68632052
sequential_1_68632054
sequential_1_68632056
sequential_1_68632058
sequential_1_68632060
sequential_1_68632062
sequential_1_68632064
sequential_1_68632066
sequential_1_68632068
sequential_1_68632070
sequential_1_68632072
sequential_1_68632074
identityѕб"sequential/StatefulPartitionedCallб$sequential_1/StatefulPartitionedCallћ
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_68632017sequential_68632019sequential_68632021sequential_68632023sequential_68632025sequential_68632027sequential_68632029sequential_68632031sequential_68632033sequential_68632035sequential_68632037sequential_68632039sequential_68632041sequential_68632043sequential_68632045sequential_68632047sequential_68632049*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *-
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_686306722$
"sequential/StatefulPartitionedCallл
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_68632052sequential_1_68632054sequential_1_68632056sequential_1_68632058sequential_1_68632060sequential_1_68632062sequential_1_68632064sequential_1_68632066sequential_1_68632068sequential_1_68632070sequential_1_68632072sequential_1_68632074*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686318052&
$sequential_1/StatefulPartitionedCall═
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Л
j
1__inference_gaussian_noise_layer_call_fn_68633939

inputs
identityѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_686310142
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ј
б
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632205

inputs
sequential_68632144
sequential_68632146
sequential_68632148
sequential_68632150
sequential_68632152
sequential_68632154
sequential_68632156
sequential_68632158
sequential_68632160
sequential_68632162
sequential_68632164
sequential_68632166
sequential_68632168
sequential_68632170
sequential_68632172
sequential_68632174
sequential_68632176
sequential_1_68632179
sequential_1_68632181
sequential_1_68632183
sequential_1_68632185
sequential_1_68632187
sequential_1_68632189
sequential_1_68632191
sequential_1_68632193
sequential_1_68632195
sequential_1_68632197
sequential_1_68632199
sequential_1_68632201
identityѕб"sequential/StatefulPartitionedCallб$sequential_1/StatefulPartitionedCallџ
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_68632144sequential_68632146sequential_68632148sequential_68632150sequential_68632152sequential_68632154sequential_68632156sequential_68632158sequential_68632160sequential_68632162sequential_68632164sequential_68632166sequential_68632168sequential_68632170sequential_68632172sequential_68632174sequential_68632176*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_686307622$
"sequential/StatefulPartitionedCallл
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_68632179sequential_1_68632181sequential_1_68632183sequential_1_68632185sequential_1_68632187sequential_1_68632189sequential_1_68632191sequential_1_68632193sequential_1_68632195sequential_1_68632197sequential_1_68632199sequential_1_68632201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686318622&
$sequential_1/StatefulPartitionedCall═
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Є
Ш
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68631218

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¤
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
д
Ф
8__inference_batch_normalization_3_layer_call_fn_68634007

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_686308572
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ї
L
0__inference_leaky_re_lu_2_layer_call_fn_68633892

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_686305232
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Є
Ш
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68631200

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¤
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
я
Ф
8__inference_batch_normalization_3_layer_call_fn_68634082

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_686310852
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ш
ъ
C__inference_dense_layer_call_and_return_conditional_losses_68633599

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђb*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђb2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*+
_input_shapes
:         ђ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Я7
д
J__inference_sequential_1_layer_call_and_return_conditional_losses_68631339
gaussian_noise_input
conv2d_68631047"
batch_normalization_3_68631112"
batch_normalization_3_68631114"
batch_normalization_3_68631116"
batch_normalization_3_68631118
conv2d_1_68631180"
batch_normalization_4_68631245"
batch_normalization_4_68631247"
batch_normalization_4_68631249"
batch_normalization_4_68631251
dense_1_68631333
dense_1_68631335
identityѕб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб!dropout_2/StatefulPartitionedCallб!dropout_3/StatefulPartitionedCallб&gaussian_noise/StatefulPartitionedCallЎ
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCallgaussian_noise_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_686310142(
&gaussian_noise/StatefulPartitionedCall▒
conv2d/StatefulPartitionedCallStatefulPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0conv2d_68631047*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_686310382 
conv2d/StatefulPartitionedCall╦
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3_68631112batch_normalization_3_68631114batch_normalization_3_68631116batch_normalization_3_68631118*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_686310672/
-batch_normalization_3/StatefulPartitionedCallа
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_686311262
leaky_re_lu_3/PartitionedCall┼
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0'^gaussian_noise/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_686311462#
!dropout_2/StatefulPartitionedCallх
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_1_68631180*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_686311712"
 conv2d_1/StatefulPartitionedCall╬
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_4_68631245batch_normalization_4_68631247batch_normalization_4_68631249batch_normalization_4_68631251*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_686312002/
-batch_normalization_4/StatefulPartitionedCallЛ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_686312662#
!dropout_3/StatefulPartitionedCallЋ
leaky_re_lu_4/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_686312892
leaky_re_lu_4/PartitionedCallэ
flatten/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_686313032
flatten/PartitionedCall▓
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_68631333dense_1_68631335*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_686313222!
dense_1/StatefulPartitionedCall│
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall'^gaussian_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall:e a
/
_output_shapes
:         
.
_user_specified_namegaussian_noise_input
г
г
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632011
sequential_input
sequential_68631950
sequential_68631952
sequential_68631954
sequential_68631956
sequential_68631958
sequential_68631960
sequential_68631962
sequential_68631964
sequential_68631966
sequential_68631968
sequential_68631970
sequential_68631972
sequential_68631974
sequential_68631976
sequential_68631978
sequential_68631980
sequential_68631982
sequential_1_68631985
sequential_1_68631987
sequential_1_68631989
sequential_1_68631991
sequential_1_68631993
sequential_1_68631995
sequential_1_68631997
sequential_1_68631999
sequential_1_68632001
sequential_1_68632003
sequential_1_68632005
sequential_1_68632007
identityѕб"sequential/StatefulPartitionedCallб$sequential_1/StatefulPartitionedCallц
"sequential/StatefulPartitionedCallStatefulPartitionedCallsequential_inputsequential_68631950sequential_68631952sequential_68631954sequential_68631956sequential_68631958sequential_68631960sequential_68631962sequential_68631964sequential_68631966sequential_68631968sequential_68631970sequential_68631972sequential_68631974sequential_68631976sequential_68631978sequential_68631980sequential_68631982*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_686307622$
"sequential/StatefulPartitionedCallл
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_68631985sequential_1_68631987sequential_1_68631989sequential_1_68631991sequential_1_68631993sequential_1_68631995sequential_1_68631997sequential_1_68631999sequential_1_68632001sequential_1_68632003sequential_1_68632005sequential_1_68632007*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686318622&
$sequential_1/StatefulPartitionedCall═
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Z V
(
_output_shapes
:         ђ
*
_user_specified_namesequential_input
и 
┐
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_68630149

inputs,
(conv2d_transpose_readvariableop_resource
identityѕбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3ѓ
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3┤
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02!
conv2d_transpose/ReadVariableOp­
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
conv2d_transposeЕ
IdentityIdentityconv2d_transpose:output:0 ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           ђ:2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ч
Ш
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68631085

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ЗD
ъ
H__inference_sequential_layer_call_and_return_conditional_losses_68630672

inputs
dense_68630624 
batch_normalization_68630627 
batch_normalization_68630629 
batch_normalization_68630631 
batch_normalization_68630633
conv2d_transpose_68630638"
batch_normalization_1_68630641"
batch_normalization_1_68630643"
batch_normalization_1_68630645"
batch_normalization_1_68630647
conv2d_transpose_1_68630652"
batch_normalization_2_68630655"
batch_normalization_2_68630657"
batch_normalization_2_68630659"
batch_normalization_2_68630661
conv2d_transpose_2_68630666
conv2d_transpose_2_68630668
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallбdense/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_68630624*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_686303172
dense/StatefulPartitionedCall│
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_68630627batch_normalization_68630629batch_normalization_68630631batch_normalization_68630633*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_686299312-
+batch_normalization/StatefulPartitionedCallЉ
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_686303692
leaky_re_lu/PartitionedCall§
reshape/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_686303912
reshape/PartitionedCallП
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_68630638*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_686300062*
(conv2d_transpose/StatefulPartitionedCallТ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_1_68630641batch_normalization_1_68630643batch_normalization_1_68630645batch_normalization_1_68630647*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_686300762/
-batch_normalization_1/StatefulPartitionedCall│
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_686304422
leaky_re_lu_1/PartitionedCallЕ
dropout/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_686304622!
dropout/StatefulPartitionedCallВ
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_transpose_1_68630652*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_686301492,
*conv2d_transpose_1/StatefulPartitionedCallу
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_2_68630655batch_normalization_2_68630657batch_normalization_2_68630659batch_normalization_2_68630661*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_686302192/
-batch_normalization_2/StatefulPartitionedCall▓
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_686305232
leaky_re_lu_2/PartitionedCallл
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_686305432#
!dropout_1/StatefulPartitionedCallЇ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_transpose_2_68630666conv2d_transpose_2_68630668*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_686302962,
*conv2d_transpose_2/StatefulPartitionedCallџ
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
У	
Е
/__inference_sequential_1_layer_call_fn_68631515
gaussian_noise_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallgaussian_noise_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686314882
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:         
.
_user_specified_namegaussian_noise_input
Й	
Џ
/__inference_sequential_1_layer_call_fn_68633563

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686314202
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┬
Ъ
D__inference_conv2d_layer_call_and_return_conditional_losses_68633951

inputs"
conv2d_readvariableop_resource
identityѕбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2DЃ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
├
Ш
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68630888

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
З
g
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_68634087

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%џЎЎ>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
К
А
F__inference_conv2d_1_layer_call_and_return_conditional_losses_68631171

inputs"
conv2d_readvariableop_resource
identityѕбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2Dё
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         @:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Е═
░#
#__inference__wrapped_model_68629835
sequential_input@
<sequential_2_sequential_dense_matmul_readvariableop_resourceQ
Msequential_2_sequential_batch_normalization_batchnorm_readvariableop_resourceU
Qsequential_2_sequential_batch_normalization_batchnorm_mul_readvariableop_resourceS
Osequential_2_sequential_batch_normalization_batchnorm_readvariableop_1_resourceS
Osequential_2_sequential_batch_normalization_batchnorm_readvariableop_2_resourceU
Qsequential_2_sequential_conv2d_transpose_conv2d_transpose_readvariableop_resourceI
Esequential_2_sequential_batch_normalization_1_readvariableop_resourceK
Gsequential_2_sequential_batch_normalization_1_readvariableop_1_resourceZ
Vsequential_2_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource\
Xsequential_2_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceW
Ssequential_2_sequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceI
Esequential_2_sequential_batch_normalization_2_readvariableop_resourceK
Gsequential_2_sequential_batch_normalization_2_readvariableop_1_resourceZ
Vsequential_2_sequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource\
Xsequential_2_sequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceW
Ssequential_2_sequential_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceN
Jsequential_2_sequential_conv2d_transpose_2_biasadd_readvariableop_resourceC
?sequential_2_sequential_1_conv2d_conv2d_readvariableop_resourceK
Gsequential_2_sequential_1_batch_normalization_3_readvariableop_resourceM
Isequential_2_sequential_1_batch_normalization_3_readvariableop_1_resource\
Xsequential_2_sequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource^
Zsequential_2_sequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceE
Asequential_2_sequential_1_conv2d_1_conv2d_readvariableop_resourceK
Gsequential_2_sequential_1_batch_normalization_4_readvariableop_resourceM
Isequential_2_sequential_1_batch_normalization_4_readvariableop_1_resource\
Xsequential_2_sequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource^
Zsequential_2_sequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceD
@sequential_2_sequential_1_dense_1_matmul_readvariableop_resourceE
Asequential_2_sequential_1_dense_1_biasadd_readvariableop_resource
identityѕбDsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOpбFsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_1бFsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_2бHsequential_2/sequential/batch_normalization/batchnorm/mul/ReadVariableOpбMsequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбOsequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б<sequential_2/sequential/batch_normalization_1/ReadVariableOpб>sequential_2/sequential/batch_normalization_1/ReadVariableOp_1бMsequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбOsequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б<sequential_2/sequential/batch_normalization_2/ReadVariableOpб>sequential_2/sequential/batch_normalization_2/ReadVariableOp_1бHsequential_2/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpбJsequential_2/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpбAsequential_2/sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpбJsequential_2/sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpб3sequential_2/sequential/dense/MatMul/ReadVariableOpбOsequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбQsequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б>sequential_2/sequential_1/batch_normalization_3/ReadVariableOpб@sequential_2/sequential_1/batch_normalization_3/ReadVariableOp_1бOsequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбQsequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б>sequential_2/sequential_1/batch_normalization_4/ReadVariableOpб@sequential_2/sequential_1/batch_normalization_4/ReadVariableOp_1б6sequential_2/sequential_1/conv2d/Conv2D/ReadVariableOpб8sequential_2/sequential_1/conv2d_1/Conv2D/ReadVariableOpб8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOpб7sequential_2/sequential_1/dense_1/MatMul/ReadVariableOpж
3sequential_2/sequential/dense/MatMul/ReadVariableOpReadVariableOp<sequential_2_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђb*
dtype025
3sequential_2/sequential/dense/MatMul/ReadVariableOpп
$sequential_2/sequential/dense/MatMulMatMulsequential_input;sequential_2/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђb2&
$sequential_2/sequential/dense/MatMulЌ
Dsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOpReadVariableOpMsequential_2_sequential_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:ђb*
dtype02F
Dsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp┐
;sequential_2/sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2=
;sequential_2/sequential/batch_normalization/batchnorm/add/y╣
9sequential_2/sequential/batch_normalization/batchnorm/addAddV2Lsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp:value:0Dsequential_2/sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђb2;
9sequential_2/sequential/batch_normalization/batchnorm/addУ
;sequential_2/sequential/batch_normalization/batchnorm/RsqrtRsqrt=sequential_2/sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ђb2=
;sequential_2/sequential/batch_normalization/batchnorm/RsqrtБ
Hsequential_2/sequential/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpQsequential_2_sequential_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђb*
dtype02J
Hsequential_2/sequential/batch_normalization/batchnorm/mul/ReadVariableOpХ
9sequential_2/sequential/batch_normalization/batchnorm/mulMul?sequential_2/sequential/batch_normalization/batchnorm/Rsqrt:y:0Psequential_2/sequential/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђb2;
9sequential_2/sequential/batch_normalization/batchnorm/mulБ
;sequential_2/sequential/batch_normalization/batchnorm/mul_1Mul.sequential_2/sequential/dense/MatMul:product:0=sequential_2/sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђb2=
;sequential_2/sequential/batch_normalization/batchnorm/mul_1Ю
Fsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpOsequential_2_sequential_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђb*
dtype02H
Fsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_1Х
;sequential_2/sequential/batch_normalization/batchnorm/mul_2MulNsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_1:value:0=sequential_2/sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђb2=
;sequential_2/sequential/batch_normalization/batchnorm/mul_2Ю
Fsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpOsequential_2_sequential_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђb*
dtype02H
Fsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_2┤
9sequential_2/sequential/batch_normalization/batchnorm/subSubNsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_2:value:0?sequential_2/sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђb2;
9sequential_2/sequential/batch_normalization/batchnorm/subХ
;sequential_2/sequential/batch_normalization/batchnorm/add_1AddV2?sequential_2/sequential/batch_normalization/batchnorm/mul_1:z:0=sequential_2/sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђb2=
;sequential_2/sequential/batch_normalization/batchnorm/add_1Т
-sequential_2/sequential/leaky_re_lu/LeakyRelu	LeakyRelu?sequential_2/sequential/batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         ђb*
alpha%џЎЎ>2/
-sequential_2/sequential/leaky_re_lu/LeakyRelu╣
%sequential_2/sequential/reshape/ShapeShape;sequential_2/sequential/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2'
%sequential_2/sequential/reshape/Shape┤
3sequential_2/sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_2/sequential/reshape/strided_slice/stackИ
5sequential_2/sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_2/sequential/reshape/strided_slice/stack_1И
5sequential_2/sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_2/sequential/reshape/strided_slice/stack_2б
-sequential_2/sequential/reshape/strided_sliceStridedSlice.sequential_2/sequential/reshape/Shape:output:0<sequential_2/sequential/reshape/strided_slice/stack:output:0>sequential_2/sequential/reshape/strided_slice/stack_1:output:0>sequential_2/sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_2/sequential/reshape/strided_sliceц
/sequential_2/sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential_2/sequential/reshape/Reshape/shape/1ц
/sequential_2/sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :21
/sequential_2/sequential/reshape/Reshape/shape/2Ц
/sequential_2/sequential/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ21
/sequential_2/sequential/reshape/Reshape/shape/3Щ
-sequential_2/sequential/reshape/Reshape/shapePack6sequential_2/sequential/reshape/strided_slice:output:08sequential_2/sequential/reshape/Reshape/shape/1:output:08sequential_2/sequential/reshape/Reshape/shape/2:output:08sequential_2/sequential/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2/
-sequential_2/sequential/reshape/Reshape/shapeЇ
'sequential_2/sequential/reshape/ReshapeReshape;sequential_2/sequential/leaky_re_lu/LeakyRelu:activations:06sequential_2/sequential/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђ2)
'sequential_2/sequential/reshape/Reshape└
.sequential_2/sequential/conv2d_transpose/ShapeShape0sequential_2/sequential/reshape/Reshape:output:0*
T0*
_output_shapes
:20
.sequential_2/sequential/conv2d_transpose/Shapeк
<sequential_2/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_2/sequential/conv2d_transpose/strided_slice/stack╩
>sequential_2/sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_2/sequential/conv2d_transpose/strided_slice/stack_1╩
>sequential_2/sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_2/sequential/conv2d_transpose/strided_slice/stack_2п
6sequential_2/sequential/conv2d_transpose/strided_sliceStridedSlice7sequential_2/sequential/conv2d_transpose/Shape:output:0Esequential_2/sequential/conv2d_transpose/strided_slice/stack:output:0Gsequential_2/sequential/conv2d_transpose/strided_slice/stack_1:output:0Gsequential_2/sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_2/sequential/conv2d_transpose/strided_sliceд
0sequential_2/sequential/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :22
0sequential_2/sequential/conv2d_transpose/stack/1д
0sequential_2/sequential/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :22
0sequential_2/sequential/conv2d_transpose/stack/2Д
0sequential_2/sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ22
0sequential_2/sequential/conv2d_transpose/stack/3ѕ
.sequential_2/sequential/conv2d_transpose/stackPack?sequential_2/sequential/conv2d_transpose/strided_slice:output:09sequential_2/sequential/conv2d_transpose/stack/1:output:09sequential_2/sequential/conv2d_transpose/stack/2:output:09sequential_2/sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:20
.sequential_2/sequential/conv2d_transpose/stack╩
>sequential_2/sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_2/sequential/conv2d_transpose/strided_slice_1/stack╬
@sequential_2/sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential/conv2d_transpose/strided_slice_1/stack_1╬
@sequential_2/sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential/conv2d_transpose/strided_slice_1/stack_2Р
8sequential_2/sequential/conv2d_transpose/strided_slice_1StridedSlice7sequential_2/sequential/conv2d_transpose/stack:output:0Gsequential_2/sequential/conv2d_transpose/strided_slice_1/stack:output:0Isequential_2/sequential/conv2d_transpose/strided_slice_1/stack_1:output:0Isequential_2/sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_2/sequential/conv2d_transpose/strided_slice_1░
Hsequential_2/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpQsequential_2_sequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02J
Hsequential_2/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpГ
9sequential_2/sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput7sequential_2/sequential/conv2d_transpose/stack:output:0Psequential_2/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:00sequential_2/sequential/reshape/Reshape:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2;
9sequential_2/sequential/conv2d_transpose/conv2d_transpose 
<sequential_2/sequential/batch_normalization_1/ReadVariableOpReadVariableOpEsequential_2_sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02>
<sequential_2/sequential/batch_normalization_1/ReadVariableOpЁ
>sequential_2/sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOpGsequential_2_sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02@
>sequential_2/sequential/batch_normalization_1/ReadVariableOp_1▓
Msequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpVsequential_2_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02O
Msequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpИ
Osequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXsequential_2_sequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02Q
Osequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ъ
>sequential_2/sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3Bsequential_2/sequential/conv2d_transpose/conv2d_transpose:output:0Dsequential_2/sequential/batch_normalization_1/ReadVariableOp:value:0Fsequential_2/sequential/batch_normalization_1/ReadVariableOp_1:value:0Usequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Wsequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2@
>sequential_2/sequential/batch_normalization_1/FusedBatchNormV3ш
/sequential_2/sequential/leaky_re_lu_1/LeakyRelu	LeakyReluBsequential_2/sequential/batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>21
/sequential_2/sequential/leaky_re_lu_1/LeakyRelu┌
(sequential_2/sequential/dropout/IdentityIdentity=sequential_2/sequential/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:         ђ2*
(sequential_2/sequential/dropout/Identity┼
0sequential_2/sequential/conv2d_transpose_1/ShapeShape1sequential_2/sequential/dropout/Identity:output:0*
T0*
_output_shapes
:22
0sequential_2/sequential/conv2d_transpose_1/Shape╩
>sequential_2/sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_2/sequential/conv2d_transpose_1/strided_slice/stack╬
@sequential_2/sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential/conv2d_transpose_1/strided_slice/stack_1╬
@sequential_2/sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential/conv2d_transpose_1/strided_slice/stack_2С
8sequential_2/sequential/conv2d_transpose_1/strided_sliceStridedSlice9sequential_2/sequential/conv2d_transpose_1/Shape:output:0Gsequential_2/sequential/conv2d_transpose_1/strided_slice/stack:output:0Isequential_2/sequential/conv2d_transpose_1/strided_slice/stack_1:output:0Isequential_2/sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_2/sequential/conv2d_transpose_1/strided_sliceф
2sequential_2/sequential/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/sequential/conv2d_transpose_1/stack/1ф
2sequential_2/sequential/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/sequential/conv2d_transpose_1/stack/2ф
2sequential_2/sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@24
2sequential_2/sequential/conv2d_transpose_1/stack/3ћ
0sequential_2/sequential/conv2d_transpose_1/stackPackAsequential_2/sequential/conv2d_transpose_1/strided_slice:output:0;sequential_2/sequential/conv2d_transpose_1/stack/1:output:0;sequential_2/sequential/conv2d_transpose_1/stack/2:output:0;sequential_2/sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:22
0sequential_2/sequential/conv2d_transpose_1/stack╬
@sequential_2/sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_2/sequential/conv2d_transpose_1/strided_slice_1/stackм
Bsequential_2/sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential/conv2d_transpose_1/strided_slice_1/stack_1м
Bsequential_2/sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential/conv2d_transpose_1/strided_slice_1/stack_2Ь
:sequential_2/sequential/conv2d_transpose_1/strided_slice_1StridedSlice9sequential_2/sequential/conv2d_transpose_1/stack:output:0Isequential_2/sequential/conv2d_transpose_1/strided_slice_1/stack:output:0Ksequential_2/sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ksequential_2/sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_2/sequential/conv2d_transpose_1/strided_slice_1х
Jsequential_2/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpSsequential_2_sequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02L
Jsequential_2/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpх
;sequential_2/sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput9sequential_2/sequential/conv2d_transpose_1/stack:output:0Rsequential_2/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:01sequential_2/sequential/dropout/Identity:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2=
;sequential_2/sequential/conv2d_transpose_1/conv2d_transpose■
<sequential_2/sequential/batch_normalization_2/ReadVariableOpReadVariableOpEsequential_2_sequential_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02>
<sequential_2/sequential/batch_normalization_2/ReadVariableOpё
>sequential_2/sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOpGsequential_2_sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>sequential_2/sequential/batch_normalization_2/ReadVariableOp_1▒
Msequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpVsequential_2_sequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02O
Msequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpи
Osequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXsequential_2_sequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02Q
Osequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ю
>sequential_2/sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3Dsequential_2/sequential/conv2d_transpose_1/conv2d_transpose:output:0Dsequential_2/sequential/batch_normalization_2/ReadVariableOp:value:0Fsequential_2/sequential/batch_normalization_2/ReadVariableOp_1:value:0Usequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Wsequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2@
>sequential_2/sequential/batch_normalization_2/FusedBatchNormV3З
/sequential_2/sequential/leaky_re_lu_2/LeakyRelu	LeakyReluBsequential_2/sequential/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>21
/sequential_2/sequential/leaky_re_lu_2/LeakyReluП
*sequential_2/sequential/dropout_1/IdentityIdentity=sequential_2/sequential/leaky_re_lu_2/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2,
*sequential_2/sequential/dropout_1/IdentityК
0sequential_2/sequential/conv2d_transpose_2/ShapeShape3sequential_2/sequential/dropout_1/Identity:output:0*
T0*
_output_shapes
:22
0sequential_2/sequential/conv2d_transpose_2/Shape╩
>sequential_2/sequential/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_2/sequential/conv2d_transpose_2/strided_slice/stack╬
@sequential_2/sequential/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential/conv2d_transpose_2/strided_slice/stack_1╬
@sequential_2/sequential/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential/conv2d_transpose_2/strided_slice/stack_2С
8sequential_2/sequential/conv2d_transpose_2/strided_sliceStridedSlice9sequential_2/sequential/conv2d_transpose_2/Shape:output:0Gsequential_2/sequential/conv2d_transpose_2/strided_slice/stack:output:0Isequential_2/sequential/conv2d_transpose_2/strided_slice/stack_1:output:0Isequential_2/sequential/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_2/sequential/conv2d_transpose_2/strided_sliceф
2sequential_2/sequential/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/sequential/conv2d_transpose_2/stack/1ф
2sequential_2/sequential/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/sequential/conv2d_transpose_2/stack/2ф
2sequential_2/sequential/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/sequential/conv2d_transpose_2/stack/3ћ
0sequential_2/sequential/conv2d_transpose_2/stackPackAsequential_2/sequential/conv2d_transpose_2/strided_slice:output:0;sequential_2/sequential/conv2d_transpose_2/stack/1:output:0;sequential_2/sequential/conv2d_transpose_2/stack/2:output:0;sequential_2/sequential/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:22
0sequential_2/sequential/conv2d_transpose_2/stack╬
@sequential_2/sequential/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_2/sequential/conv2d_transpose_2/strided_slice_1/stackм
Bsequential_2/sequential/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential/conv2d_transpose_2/strided_slice_1/stack_1м
Bsequential_2/sequential/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential/conv2d_transpose_2/strided_slice_1/stack_2Ь
:sequential_2/sequential/conv2d_transpose_2/strided_slice_1StridedSlice9sequential_2/sequential/conv2d_transpose_2/stack:output:0Isequential_2/sequential/conv2d_transpose_2/strided_slice_1/stack:output:0Ksequential_2/sequential/conv2d_transpose_2/strided_slice_1/stack_1:output:0Ksequential_2/sequential/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_2/sequential/conv2d_transpose_2/strided_slice_1┤
Jsequential_2/sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpSsequential_2_sequential_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02L
Jsequential_2/sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpи
;sequential_2/sequential/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput9sequential_2/sequential/conv2d_transpose_2/stack:output:0Rsequential_2/sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:03sequential_2/sequential/dropout_1/Identity:output:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2=
;sequential_2/sequential/conv2d_transpose_2/conv2d_transposeЇ
Asequential_2/sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_sequential_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_2/sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpЙ
2sequential_2/sequential/conv2d_transpose_2/BiasAddBiasAddDsequential_2/sequential/conv2d_transpose_2/conv2d_transpose:output:0Isequential_2/sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         24
2sequential_2/sequential/conv2d_transpose_2/BiasAddр
/sequential_2/sequential/conv2d_transpose_2/TanhTanh;sequential_2/sequential/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:         21
/sequential_2/sequential/conv2d_transpose_2/TanhЭ
6sequential_2/sequential_1/conv2d/Conv2D/ReadVariableOpReadVariableOp?sequential_2_sequential_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype028
6sequential_2/sequential_1/conv2d/Conv2D/ReadVariableOp│
'sequential_2/sequential_1/conv2d/Conv2DConv2D3sequential_2/sequential/conv2d_transpose_2/Tanh:y:0>sequential_2/sequential_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2)
'sequential_2/sequential_1/conv2d/Conv2Dё
>sequential_2/sequential_1/batch_normalization_3/ReadVariableOpReadVariableOpGsequential_2_sequential_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>sequential_2/sequential_1/batch_normalization_3/ReadVariableOpі
@sequential_2/sequential_1/batch_normalization_3/ReadVariableOp_1ReadVariableOpIsequential_2_sequential_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@sequential_2/sequential_1/batch_normalization_3/ReadVariableOp_1и
Osequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpXsequential_2_sequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Q
Osequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpй
Qsequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZsequential_2_sequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02S
Qsequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ћ
@sequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV30sequential_2/sequential_1/conv2d/Conv2D:output:0Fsequential_2/sequential_1/batch_normalization_3/ReadVariableOp:value:0Hsequential_2/sequential_1/batch_normalization_3/ReadVariableOp_1:value:0Wsequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Ysequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2B
@sequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3Щ
1sequential_2/sequential_1/leaky_re_lu_3/LeakyRelu	LeakyReluDsequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>23
1sequential_2/sequential_1/leaky_re_lu_3/LeakyReluс
,sequential_2/sequential_1/dropout_2/IdentityIdentity?sequential_2/sequential_1/leaky_re_lu_3/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2.
,sequential_2/sequential_1/dropout_2/Identity 
8sequential_2/sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpAsequential_2_sequential_1_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02:
8sequential_2/sequential_1/conv2d_1/Conv2D/ReadVariableOp╝
)sequential_2/sequential_1/conv2d_1/Conv2DConv2D5sequential_2/sequential_1/dropout_2/Identity:output:0@sequential_2/sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2+
)sequential_2/sequential_1/conv2d_1/Conv2DЁ
>sequential_2/sequential_1/batch_normalization_4/ReadVariableOpReadVariableOpGsequential_2_sequential_1_batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02@
>sequential_2/sequential_1/batch_normalization_4/ReadVariableOpІ
@sequential_2/sequential_1/batch_normalization_4/ReadVariableOp_1ReadVariableOpIsequential_2_sequential_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02B
@sequential_2/sequential_1/batch_normalization_4/ReadVariableOp_1И
Osequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpXsequential_2_sequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02Q
Osequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpЙ
Qsequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZsequential_2_sequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02S
Qsequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Џ
@sequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV32sequential_2/sequential_1/conv2d_1/Conv2D:output:0Fsequential_2/sequential_1/batch_normalization_4/ReadVariableOp:value:0Hsequential_2/sequential_1/batch_normalization_4/ReadVariableOp_1:value:0Wsequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Ysequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2B
@sequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3ж
,sequential_2/sequential_1/dropout_3/IdentityIdentityDsequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ђ2.
,sequential_2/sequential_1/dropout_3/IdentityВ
1sequential_2/sequential_1/leaky_re_lu_4/LeakyRelu	LeakyRelu5sequential_2/sequential_1/dropout_3/Identity:output:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>23
1sequential_2/sequential_1/leaky_re_lu_4/LeakyReluБ
'sequential_2/sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2)
'sequential_2/sequential_1/flatten/ConstЄ
)sequential_2/sequential_1/flatten/ReshapeReshape?sequential_2/sequential_1/leaky_re_lu_4/LeakyRelu:activations:00sequential_2/sequential_1/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ12+
)sequential_2/sequential_1/flatten/ReshapeЗ
7sequential_2/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp@sequential_2_sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype029
7sequential_2/sequential_1/dense_1/MatMul/ReadVariableOpЁ
(sequential_2/sequential_1/dense_1/MatMulMatMul2sequential_2/sequential_1/flatten/Reshape:output:0?sequential_2/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2*
(sequential_2/sequential_1/dense_1/MatMulЫ
8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpAsequential_2_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOpЅ
)sequential_2/sequential_1/dense_1/BiasAddBiasAdd2sequential_2/sequential_1/dense_1/MatMul:product:0@sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2+
)sequential_2/sequential_1/dense_1/BiasAddК
)sequential_2/sequential_1/dense_1/SigmoidSigmoid2sequential_2/sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2+
)sequential_2/sequential_1/dense_1/SigmoidЁ
IdentityIdentity-sequential_2/sequential_1/dense_1/Sigmoid:y:0E^sequential_2/sequential/batch_normalization/batchnorm/ReadVariableOpG^sequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_1G^sequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_2I^sequential_2/sequential/batch_normalization/batchnorm/mul/ReadVariableOpN^sequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpP^sequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=^sequential_2/sequential/batch_normalization_1/ReadVariableOp?^sequential_2/sequential/batch_normalization_1/ReadVariableOp_1N^sequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpP^sequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=^sequential_2/sequential/batch_normalization_2/ReadVariableOp?^sequential_2/sequential/batch_normalization_2/ReadVariableOp_1I^sequential_2/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpK^sequential_2/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpB^sequential_2/sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpK^sequential_2/sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp4^sequential_2/sequential/dense/MatMul/ReadVariableOpP^sequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpR^sequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?^sequential_2/sequential_1/batch_normalization_3/ReadVariableOpA^sequential_2/sequential_1/batch_normalization_3/ReadVariableOp_1P^sequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpR^sequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?^sequential_2/sequential_1/batch_normalization_4/ReadVariableOpA^sequential_2/sequential_1/batch_normalization_4/ReadVariableOp_17^sequential_2/sequential_1/conv2d/Conv2D/ReadVariableOp9^sequential_2/sequential_1/conv2d_1/Conv2D/ReadVariableOp9^sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOp8^sequential_2/sequential_1/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::2ї
Dsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOpDsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp2љ
Fsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_1Fsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_12љ
Fsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_2Fsequential_2/sequential/batch_normalization/batchnorm/ReadVariableOp_22ћ
Hsequential_2/sequential/batch_normalization/batchnorm/mul/ReadVariableOpHsequential_2/sequential/batch_normalization/batchnorm/mul/ReadVariableOp2ъ
Msequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpMsequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2б
Osequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Osequential_2/sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12|
<sequential_2/sequential/batch_normalization_1/ReadVariableOp<sequential_2/sequential/batch_normalization_1/ReadVariableOp2ђ
>sequential_2/sequential/batch_normalization_1/ReadVariableOp_1>sequential_2/sequential/batch_normalization_1/ReadVariableOp_12ъ
Msequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpMsequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2б
Osequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Osequential_2/sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12|
<sequential_2/sequential/batch_normalization_2/ReadVariableOp<sequential_2/sequential/batch_normalization_2/ReadVariableOp2ђ
>sequential_2/sequential/batch_normalization_2/ReadVariableOp_1>sequential_2/sequential/batch_normalization_2/ReadVariableOp_12ћ
Hsequential_2/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpHsequential_2/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2ў
Jsequential_2/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpJsequential_2/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2є
Asequential_2/sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpAsequential_2/sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp2ў
Jsequential_2/sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpJsequential_2/sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2j
3sequential_2/sequential/dense/MatMul/ReadVariableOp3sequential_2/sequential/dense/MatMul/ReadVariableOp2б
Osequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpOsequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2д
Qsequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Qsequential_2/sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12ђ
>sequential_2/sequential_1/batch_normalization_3/ReadVariableOp>sequential_2/sequential_1/batch_normalization_3/ReadVariableOp2ё
@sequential_2/sequential_1/batch_normalization_3/ReadVariableOp_1@sequential_2/sequential_1/batch_normalization_3/ReadVariableOp_12б
Osequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpOsequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2д
Qsequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Qsequential_2/sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12ђ
>sequential_2/sequential_1/batch_normalization_4/ReadVariableOp>sequential_2/sequential_1/batch_normalization_4/ReadVariableOp2ё
@sequential_2/sequential_1/batch_normalization_4/ReadVariableOp_1@sequential_2/sequential_1/batch_normalization_4/ReadVariableOp_12p
6sequential_2/sequential_1/conv2d/Conv2D/ReadVariableOp6sequential_2/sequential_1/conv2d/Conv2D/ReadVariableOp2t
8sequential_2/sequential_1/conv2d_1/Conv2D/ReadVariableOp8sequential_2/sequential_1/conv2d_1/Conv2D/ReadVariableOp2t
8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOp8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOp2r
7sequential_2/sequential_1/dense_1/MatMul/ReadVariableOp7sequential_2/sequential_1/dense_1/MatMul/ReadVariableOp:Z V
(
_output_shapes
:         ђ
*
_user_specified_namesequential_input
Б
J
.__inference_leaky_re_lu_layer_call_fn_68633698

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_686303692
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђb:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
├
Ш
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68633994

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
├
Ш
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68633856

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
с
г
&__inference_signature_wrapper_68632345
sequential_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityѕбStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *?
_read_only_resource_inputs!
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference__wrapped_model_686298352
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:         ђ
*
_user_specified_namesequential_input
ф
Ф
8__inference_batch_normalization_4_layer_call_fn_68634182

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall╗
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_686309572
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
┐
a
E__inference_flatten_layer_call_and_return_conditional_losses_68634300

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ12	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ12

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Р
Ф
8__inference_batch_normalization_4_layer_call_fn_68634257

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_686312182
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
й
g
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_68630523

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           @*
alpha%џЎЎ>2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
¤
Ш
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634169

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
┐
H
,__inference_dropout_3_layer_call_fn_68634284

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_686312712
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
ћ^
Ѓ

J__inference_sequential_1_layer_call_and_return_conditional_losses_68633483

inputs)
%conv2d_conv2d_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1бconv2d/Conv2D/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpb
gaussian_noise/ShapeShapeinputs*
T0*
_output_shapes
:2
gaussian_noise/ShapeІ
!gaussian_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!gaussian_noise/random_normal/meanЈ
#gaussian_noise/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2%
#gaussian_noise/random_normal/stddevЃ
1gaussian_noise/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise/Shape:output:0*
T0*/
_output_shapes
:         *
dtype0*
seed▒ т)*
seed2║Я▓23
1gaussian_noise/random_normal/RandomStandardNormal№
 gaussian_noise/random_normal/mulMul:gaussian_noise/random_normal/RandomStandardNormal:output:0,gaussian_noise/random_normal/stddev:output:0*
T0*/
_output_shapes
:         2"
 gaussian_noise/random_normal/mul¤
gaussian_noise/random_normalAdd$gaussian_noise/random_normal/mul:z:0*gaussian_noise/random_normal/mean:output:0*
T0*/
_output_shapes
:         2
gaussian_noise/random_normalЋ
gaussian_noise/addAddV2inputs gaussian_noise/random_normal:z:0*
T0*/
_output_shapes
:         2
gaussian_noise/addф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╚
conv2d/Conv2DConv2Dgaussian_noise/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d/Conv2DХ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1я
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3г
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>2
leaky_re_lu_3/LeakyReluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout_2/dropout/ConstИ
dropout_2/dropout/MulMul%leaky_re_lu_3/LeakyRelu:activations:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout_2/dropout/MulЄ
dropout_2/dropout/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape┌
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЅ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2"
 dropout_2/dropout/GreaterEqual/yЬ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2 
dropout_2/dropout/GreaterEqualЦ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout_2/dropout/Castф
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout_2/dropout/Mul_1▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpн
conv2d_1/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_1/Conv2Dи
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_4/ReadVariableOpй
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_4/ReadVariableOp_1Ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1т
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout_3/dropout/ConstЙ
dropout_3/dropout/MulMul*batch_normalization_4/FusedBatchNormV3:y:0 dropout_3/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout_3/dropout/Mulї
dropout_3/dropout/ShapeShape*batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape█
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЅ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2"
 dropout_3/dropout/GreaterEqual/y№
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2 
dropout_3/dropout/GreaterEqualд
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout_3/dropout/CastФ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout_3/dropout/Mul_1ъ
leaky_re_lu_4/LeakyRelu	LeakyReludropout_3/dropout/Mul_1:z:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>2
leaky_re_lu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
flatten/ConstЪ
flatten/ReshapeReshape%leaky_re_lu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ12
flatten/Reshapeд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/SigmoidВ
IdentityIdentitydense_1/Sigmoid:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
о
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_68630369

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         ђb*
alpha%џЎЎ>2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђb:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
¤
џ
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68630219

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3Г
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
я
Ф
8__inference_batch_normalization_3_layer_call_fn_68634069

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_686310672
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Й
Е
6__inference_batch_normalization_layer_call_fn_68633688

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_686299642
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђb::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
К
А
F__inference_conv2d_1_layer_call_and_return_conditional_losses_68634126

inputs"
conv2d_readvariableop_resource
identityѕбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2Dё
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         @:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
§ў
█
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632730

inputs3
/sequential_dense_matmul_readvariableop_resourceD
@sequential_batch_normalization_batchnorm_readvariableop_resourceH
Dsequential_batch_normalization_batchnorm_mul_readvariableop_resourceF
Bsequential_batch_normalization_batchnorm_readvariableop_1_resourceF
Bsequential_batch_normalization_batchnorm_readvariableop_2_resourceH
Dsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource<
8sequential_batch_normalization_1_readvariableop_resource>
:sequential_batch_normalization_1_readvariableop_1_resourceM
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceJ
Fsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource<
8sequential_batch_normalization_2_readvariableop_resource>
:sequential_batch_normalization_2_readvariableop_1_resourceM
Isequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceJ
Fsequential_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceA
=sequential_conv2d_transpose_2_biasadd_readvariableop_resource6
2sequential_1_conv2d_conv2d_readvariableop_resource>
:sequential_1_batch_normalization_3_readvariableop_resource@
<sequential_1_batch_normalization_3_readvariableop_1_resourceO
Ksequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource8
4sequential_1_conv2d_1_conv2d_readvariableop_resource>
:sequential_1_batch_normalization_4_readvariableop_resource@
<sequential_1_batch_normalization_4_readvariableop_1_resourceO
Ksequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identityѕб7sequential/batch_normalization/batchnorm/ReadVariableOpб9sequential/batch_normalization/batchnorm/ReadVariableOp_1б9sequential/batch_normalization/batchnorm/ReadVariableOp_2б;sequential/batch_normalization/batchnorm/mul/ReadVariableOpб@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбBsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б/sequential/batch_normalization_1/ReadVariableOpб1sequential/batch_normalization_1/ReadVariableOp_1б@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбBsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б/sequential/batch_normalization_2/ReadVariableOpб1sequential/batch_normalization_2/ReadVariableOp_1б;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpб=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpб4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpб=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpб&sequential/dense/MatMul/ReadVariableOpбBsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбDsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б1sequential_1/batch_normalization_3/ReadVariableOpб3sequential_1/batch_normalization_3/ReadVariableOp_1бBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_1/batch_normalization_4/ReadVariableOpб3sequential_1/batch_normalization_4/ReadVariableOp_1б)sequential_1/conv2d/Conv2D/ReadVariableOpб+sequential_1/conv2d_1/Conv2D/ReadVariableOpб+sequential_1/dense_1/BiasAdd/ReadVariableOpб*sequential_1/dense_1/MatMul/ReadVariableOp┬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђb*
dtype02(
&sequential/dense/MatMul/ReadVariableOpД
sequential/dense/MatMulMatMulinputs.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђb2
sequential/dense/MatMul­
7sequential/batch_normalization/batchnorm/ReadVariableOpReadVariableOp@sequential_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:ђb*
dtype029
7sequential/batch_normalization/batchnorm/ReadVariableOpЦ
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:20
.sequential/batch_normalization/batchnorm/add/yЁ
,sequential/batch_normalization/batchnorm/addAddV2?sequential/batch_normalization/batchnorm/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђb2.
,sequential/batch_normalization/batchnorm/add┴
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ђb20
.sequential/batch_normalization/batchnorm/RsqrtЧ
;sequential/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpDsequential_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђb*
dtype02=
;sequential/batch_normalization/batchnorm/mul/ReadVariableOpѓ
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0Csequential/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђb2.
,sequential/batch_normalization/batchnorm/mul№
.sequential/batch_normalization/batchnorm/mul_1Mul!sequential/dense/MatMul:product:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђb20
.sequential/batch_normalization/batchnorm/mul_1Ш
9sequential/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpBsequential_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђb*
dtype02;
9sequential/batch_normalization/batchnorm/ReadVariableOp_1ѓ
.sequential/batch_normalization/batchnorm/mul_2MulAsequential/batch_normalization/batchnorm/ReadVariableOp_1:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђb20
.sequential/batch_normalization/batchnorm/mul_2Ш
9sequential/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpBsequential_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђb*
dtype02;
9sequential/batch_normalization/batchnorm/ReadVariableOp_2ђ
,sequential/batch_normalization/batchnorm/subSubAsequential/batch_normalization/batchnorm/ReadVariableOp_2:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђb2.
,sequential/batch_normalization/batchnorm/subѓ
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђb20
.sequential/batch_normalization/batchnorm/add_1┐
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu2sequential/batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         ђb*
alpha%џЎЎ>2"
 sequential/leaky_re_lu/LeakyReluњ
sequential/reshape/ShapeShape.sequential/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
sequential/reshape/Shapeџ
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/reshape/strided_slice/stackъ
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_1ъ
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/reshape/strided_slice/stack_2н
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential/reshape/strided_sliceі
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/1і
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"sequential/reshape/Reshape/shape/2І
"sequential/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2$
"sequential/reshape/Reshape/shape/3г
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0+sequential/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 sequential/reshape/Reshape/shape┘
sequential/reshape/ReshapeReshape.sequential/leaky_re_lu/LeakyRelu:activations:0)sequential/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђ2
sequential/reshape/ReshapeЎ
!sequential/conv2d_transpose/ShapeShape#sequential/reshape/Reshape:output:0*
T0*
_output_shapes
:2#
!sequential/conv2d_transpose/Shapeг
/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/sequential/conv2d_transpose/strided_slice/stack░
1sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice/stack_1░
1sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1sequential/conv2d_transpose/strided_slice/stack_2і
)sequential/conv2d_transpose/strided_sliceStridedSlice*sequential/conv2d_transpose/Shape:output:08sequential/conv2d_transpose/strided_slice/stack:output:0:sequential/conv2d_transpose/strided_slice/stack_1:output:0:sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)sequential/conv2d_transpose/strided_sliceї
#sequential/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/conv2d_transpose/stack/1ї
#sequential/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/conv2d_transpose/stack/2Ї
#sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2%
#sequential/conv2d_transpose/stack/3║
!sequential/conv2d_transpose/stackPack2sequential/conv2d_transpose/strided_slice:output:0,sequential/conv2d_transpose/stack/1:output:0,sequential/conv2d_transpose/stack/2:output:0,sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!sequential/conv2d_transpose/stack░
1sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose/strided_slice_1/stack┤
3sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_1/stack_1┤
3sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose/strided_slice_1/stack_2ћ
+sequential/conv2d_transpose/strided_slice_1StridedSlice*sequential/conv2d_transpose/stack:output:0:sequential/conv2d_transpose/strided_slice_1/stack:output:0<sequential/conv2d_transpose/strided_slice_1/stack_1:output:0<sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose/strided_slice_1Ѕ
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02=
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpВ
,sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput*sequential/conv2d_transpose/stack:output:0Csequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0#sequential/reshape/Reshape:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2.
,sequential/conv2d_transpose/conv2d_transposeп
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/sequential/batch_normalization_1/ReadVariableOpя
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1І
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpЉ
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1─
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV35sequential/conv2d_transpose/conv2d_transpose:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 23
1sequential/batch_normalization_1/FusedBatchNormV3╬
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu5sequential/batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>2$
"sequential/leaky_re_lu_1/LeakyRelu│
sequential/dropout/IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:         ђ2
sequential/dropout/Identityъ
#sequential/conv2d_transpose_1/ShapeShape$sequential/dropout/Identity:output:0*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_1/Shape░
1sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose_1/strided_slice/stack┤
3sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice/stack_1┤
3sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_1/strided_slice/stack_2ќ
+sequential/conv2d_transpose_1/strided_sliceStridedSlice,sequential/conv2d_transpose_1/Shape:output:0:sequential/conv2d_transpose_1/strided_slice/stack:output:0<sequential/conv2d_transpose_1/strided_slice/stack_1:output:0<sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose_1/strided_sliceљ
%sequential/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_1/stack/1љ
%sequential/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_1/stack/2љ
%sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential/conv2d_transpose_1/stack/3к
#sequential/conv2d_transpose_1/stackPack4sequential/conv2d_transpose_1/strided_slice:output:0.sequential/conv2d_transpose_1/stack/1:output:0.sequential/conv2d_transpose_1/stack/2:output:0.sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_1/stack┤
3sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/conv2d_transpose_1/strided_slice_1/stackИ
5sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_1/stack_1И
5sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_1/strided_slice_1/stack_2а
-sequential/conv2d_transpose_1/strided_slice_1StridedSlice,sequential/conv2d_transpose_1/stack:output:0<sequential/conv2d_transpose_1/strided_slice_1/stack:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_1/strided_slice_1ј
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02?
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpЗ
.sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_1/stack:output:0Esequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0$sequential/dropout/Identity:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
20
.sequential/conv2d_transpose_1/conv2d_transposeО
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential/batch_normalization_2/ReadVariableOpП
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1sequential/batch_normalization_2/ReadVariableOp_1і
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpљ
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1┴
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV37sequential/conv2d_transpose_1/conv2d_transpose:output:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 23
1sequential/batch_normalization_2/FusedBatchNormV3═
"sequential/leaky_re_lu_2/LeakyRelu	LeakyRelu5sequential/batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>2$
"sequential/leaky_re_lu_2/LeakyReluХ
sequential/dropout_1/IdentityIdentity0sequential/leaky_re_lu_2/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2
sequential/dropout_1/Identityа
#sequential/conv2d_transpose_2/ShapeShape&sequential/dropout_1/Identity:output:0*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_2/Shape░
1sequential/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential/conv2d_transpose_2/strided_slice/stack┤
3sequential/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_2/strided_slice/stack_1┤
3sequential/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential/conv2d_transpose_2/strided_slice/stack_2ќ
+sequential/conv2d_transpose_2/strided_sliceStridedSlice,sequential/conv2d_transpose_2/Shape:output:0:sequential/conv2d_transpose_2/strided_slice/stack:output:0<sequential/conv2d_transpose_2/strided_slice/stack_1:output:0<sequential/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential/conv2d_transpose_2/strided_sliceљ
%sequential/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_2/stack/1љ
%sequential/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_2/stack/2љ
%sequential/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/conv2d_transpose_2/stack/3к
#sequential/conv2d_transpose_2/stackPack4sequential/conv2d_transpose_2/strided_slice:output:0.sequential/conv2d_transpose_2/stack/1:output:0.sequential/conv2d_transpose_2/stack/2:output:0.sequential/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential/conv2d_transpose_2/stack┤
3sequential/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/conv2d_transpose_2/strided_slice_1/stackИ
5sequential/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_2/strided_slice_1/stack_1И
5sequential/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/conv2d_transpose_2/strided_slice_1/stack_2а
-sequential/conv2d_transpose_2/strided_slice_1StridedSlice,sequential/conv2d_transpose_2/stack:output:0<sequential/conv2d_transpose_2/strided_slice_1/stack:output:0>sequential/conv2d_transpose_2/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential/conv2d_transpose_2/strided_slice_1Ї
=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02?
=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOpШ
.sequential/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_2/stack:output:0Esequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0&sequential/dropout_1/Identity:output:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
20
.sequential/conv2d_transpose_2/conv2d_transposeТ
4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp=sequential_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOpі
%sequential/conv2d_transpose_2/BiasAddBiasAdd7sequential/conv2d_transpose_2/conv2d_transpose:output:0<sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2'
%sequential/conv2d_transpose_2/BiasAdd║
"sequential/conv2d_transpose_2/TanhTanh.sequential/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:         2$
"sequential/conv2d_transpose_2/TanhЛ
)sequential_1/conv2d/Conv2D/ReadVariableOpReadVariableOp2sequential_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02+
)sequential_1/conv2d/Conv2D/ReadVariableOp 
sequential_1/conv2d/Conv2DConv2D&sequential/conv2d_transpose_2/Tanh:y:01sequential_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
sequential_1/conv2d/Conv2DП
1sequential_1/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_1/batch_normalization_3/ReadVariableOpс
3sequential_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3sequential_1/batch_normalization_3/ReadVariableOp_1љ
Bsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpќ
Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1╣
3sequential_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3#sequential_1/conv2d/Conv2D:output:09sequential_1/batch_normalization_3/ReadVariableOp:value:0;sequential_1/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 25
3sequential_1/batch_normalization_3/FusedBatchNormV3М
$sequential_1/leaky_re_lu_3/LeakyRelu	LeakyRelu7sequential_1/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>2&
$sequential_1/leaky_re_lu_3/LeakyRelu╝
sequential_1/dropout_2/IdentityIdentity2sequential_1/leaky_re_lu_3/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2!
sequential_1/dropout_2/Identityп
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02-
+sequential_1/conv2d_1/Conv2D/ReadVariableOpѕ
sequential_1/conv2d_1/Conv2DConv2D(sequential_1/dropout_2/Identity:output:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
sequential_1/conv2d_1/Conv2Dя
1sequential_1/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype023
1sequential_1/batch_normalization_4/ReadVariableOpС
3sequential_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype025
3sequential_1/batch_normalization_4/ReadVariableOp_1Љ
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02D
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpЌ
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02F
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1└
3sequential_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%sequential_1/conv2d_1/Conv2D:output:09sequential_1/batch_normalization_4/ReadVariableOp:value:0;sequential_1/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 25
3sequential_1/batch_normalization_4/FusedBatchNormV3┬
sequential_1/dropout_3/IdentityIdentity7sequential_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ђ2!
sequential_1/dropout_3/Identity┼
$sequential_1/leaky_re_lu_4/LeakyRelu	LeakyRelu(sequential_1/dropout_3/Identity:output:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>2&
$sequential_1/leaky_re_lu_4/LeakyReluЅ
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
sequential_1/flatten/ConstМ
sequential_1/flatten/ReshapeReshape2sequential_1/leaky_re_lu_4/LeakyRelu:activations:0#sequential_1/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ12
sequential_1/flatten/Reshape═
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOpЛ
sequential_1/dense_1/MatMulMatMul%sequential_1/flatten/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_1/MatMul╦
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpН
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_1/BiasAddа
sequential_1/dense_1/SigmoidSigmoid%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential_1/dense_1/Sigmoid 
IdentityIdentity sequential_1/dense_1/Sigmoid:y:08^sequential/batch_normalization/batchnorm/ReadVariableOp:^sequential/batch_normalization/batchnorm/ReadVariableOp_1:^sequential/batch_normalization/batchnorm/ReadVariableOp_2<^sequential/batch_normalization/batchnorm/mul/ReadVariableOpA^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_1A^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_2/ReadVariableOp2^sequential/batch_normalization_2/ReadVariableOp_1<^sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp>^sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp5^sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp>^sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOpC^sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_3/ReadVariableOp4^sequential_1/batch_normalization_3/ReadVariableOp_1C^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_4/ReadVariableOp4^sequential_1/batch_normalization_4/ReadVariableOp_1*^sequential_1/conv2d/Conv2D/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::2r
7sequential/batch_normalization/batchnorm/ReadVariableOp7sequential/batch_normalization/batchnorm/ReadVariableOp2v
9sequential/batch_normalization/batchnorm/ReadVariableOp_19sequential/batch_normalization/batchnorm/ReadVariableOp_12v
9sequential/batch_normalization/batchnorm/ReadVariableOp_29sequential/batch_normalization/batchnorm/ReadVariableOp_22z
;sequential/batch_normalization/batchnorm/mul/ReadVariableOp;sequential/batch_normalization/batchnorm/mul/ReadVariableOp2ё
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2ѕ
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12ё
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2ѕ
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_2/ReadVariableOp/sequential/batch_normalization_2/ReadVariableOp2f
1sequential/batch_normalization_2/ReadVariableOp_11sequential/batch_normalization_2/ReadVariableOp_12z
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2~
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2l
4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp4sequential/conv2d_transpose_2/BiasAdd/ReadVariableOp2~
=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2ѕ
Bsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_3/ReadVariableOp1sequential_1/batch_normalization_3/ReadVariableOp2j
3sequential_1/batch_normalization_3/ReadVariableOp_13sequential_1/batch_normalization_3/ReadVariableOp_12ѕ
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2ї
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_4/ReadVariableOp1sequential_1/batch_normalization_4/ReadVariableOp2j
3sequential_1/batch_normalization_4/ReadVariableOp_13sequential_1/batch_normalization_4/ReadVariableOp_12V
)sequential_1/conv2d/Conv2D/ReadVariableOp)sequential_1/conv2d/Conv2D/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
¤
џ
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68633838

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3Г
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ё
H
,__inference_dropout_1_layer_call_fn_68633919

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_686305482
PartitionedCallє
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
┼2
Д
J__inference_sequential_1_layer_call_and_return_conditional_losses_68631488

inputs
conv2d_68631453"
batch_normalization_3_68631456"
batch_normalization_3_68631458"
batch_normalization_3_68631460"
batch_normalization_3_68631462
conv2d_1_68631467"
batch_normalization_4_68631470"
batch_normalization_4_68631472"
batch_normalization_4_68631474"
batch_normalization_4_68631476
dense_1_68631482
dense_1_68631484
identityѕб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense_1/StatefulPartitionedCallз
gaussian_noise/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_686310182 
gaussian_noise/PartitionedCallЕ
conv2d/StatefulPartitionedCallStatefulPartitionedCall'gaussian_noise/PartitionedCall:output:0conv2d_68631453*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_686310382 
conv2d/StatefulPartitionedCall╦
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3_68631456batch_normalization_3_68631458batch_normalization_3_68631460batch_normalization_3_68631462*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_686310852/
-batch_normalization_3/StatefulPartitionedCallа
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_686311262
leaky_re_lu_3/PartitionedCallё
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_686311512
dropout_2/PartitionedCallГ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_1_68631467*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_686311712"
 conv2d_1/StatefulPartitionedCall╬
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_4_68631470batch_normalization_4_68631472batch_normalization_4_68631474batch_normalization_4_68631476*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_686312182/
-batch_normalization_4/StatefulPartitionedCallЋ
dropout_3/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_686312712
dropout_3/PartitionedCallЇ
leaky_re_lu_4/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_686312892
leaky_re_lu_4/PartitionedCallэ
flatten/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_686313032
flatten/PartitionedCall▓
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_68631482dense_1_68631484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_686313222!
dense_1/StatefulPartitionedCall┬
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
х
c
E__inference_dropout_layer_call_and_return_conditional_losses_68633808

inputs

identity_1u
IdentityIdentityinputs*
T0*B
_output_shapes0
.:,                           ђ2

Identityё

Identity_1IdentityIdentity:output:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
т
q
+__inference_conv2d_1_layer_call_fn_68634133

inputs
unknown
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_686311712
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         @:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
¤
Ш
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68630957

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
е
Ф
8__inference_batch_normalization_1_layer_call_fn_68633768

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_686300762
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
¤
Ш
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68630107

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ё
F
*__inference_dropout_layer_call_fn_68633818

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_686304672
PartitionedCallЄ
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ЇJ
Ѓ

J__inference_sequential_1_layer_call_and_return_conditional_losses_68631862

inputs)
%conv2d_conv2d_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1бconv2d/Conv2D/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╩
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
conv2d/Conv2DХ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1­
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Й
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*A
_output_shapes/
-:+                           @*
alpha%џЎЎ>2
leaky_re_lu_3/LeakyReluД
dropout_2/IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2
dropout_2/Identity▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpТ
conv2d_1/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
2
conv2d_1/Conv2Dи
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_4/ReadVariableOpй
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_4/ReadVariableOp_1Ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3Г
dropout_3/IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout_3/Identity░
leaky_re_lu_4/LeakyRelu	LeakyReludropout_3/Identity:output:0*B
_output_shapes0
.:,                           ђ*
alpha%џЎЎ>2
leaky_re_lu_4/LeakyRelus
flatten/ShapeShape%leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
flatten/Shapeё
flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
flatten/strided_slice/stackѕ
flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
flatten/strided_slice/stack_1ѕ
flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
flatten/strided_slice/stack_2њ
flatten/strided_sliceStridedSliceflatten/Shape:output:0$flatten/strided_slice/stack:output:0&flatten/strided_slice/stack_1:output:0&flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
flatten/strided_slice}
flatten/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2
flatten/Reshape/shape/1д
flatten/Reshape/shapePackflatten/strided_slice:output:0 flatten/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
flatten/Reshape/shape»
flatten/ReshapeReshape%leaky_re_lu_4/LeakyRelu:activations:0flatten/Reshape/shape:output:0*
T0*0
_output_shapes
:                  2
flatten/Reshapeд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/SigmoidВ
IdentityIdentitydense_1/Sigmoid:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:+                           ::::::::::::2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
З	
я
E__inference_dense_1_layer_call_and_return_conditional_losses_68634316

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoidљ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ1::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ1
 
_user_specified_nameinputs
Ш
h
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_68633934

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
і
ж
-__inference_sequential_layer_call_fn_68633204

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identityѕбStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_686307622
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
о
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_68633904

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/ConstЇ
dropout/MulMulinputsdropout/Const:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/yп
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/GreaterEqualЎ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout/Castћ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
¤
Ш
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68630988

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ј
e
,__inference_dropout_1_layer_call_fn_68633914

inputs
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_686305432
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╣ 
й
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_68630006

inputs,
(conv2d_transpose_readvariableop_resource
identityѕбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2	
stack/3ѓ
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3х
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_transpose/ReadVariableOpы
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
2
conv2d_transposeф
IdentityIdentityconv2d_transpose:output:0 ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           ђ:2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ї
х
/__inference_sequential_2_layer_call_fn_68632139
sequential_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *9
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_686320782
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:         ђ
*
_user_specified_namesequential_input
Ш
h
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_68631018

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┐
y
3__inference_conv2d_transpose_layer_call_fn_68630014

inputs
unknown
identityѕбStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_686300062
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,                           ђ:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ф
F
*__inference_flatten_layer_call_fn_68634305

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_686313032
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ12

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
№
Ф
/__inference_sequential_2_layer_call_fn_68632793

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *9
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_686320782
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┴
g
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_68633786

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,                           ђ*
alpha%џЎЎ>2
	LeakyReluє
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
с

*__inference_dense_1_layer_call_fn_68634325

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_686313222
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ1::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ1
 
_user_specified_nameinputs
┼
M
1__inference_gaussian_noise_layer_call_fn_68633944

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_686310182
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
├
L
0__inference_leaky_re_lu_3_layer_call_fn_68634092

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_686311262
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ч
Ш
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68634038

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Цг
Љ
H__inference_sequential_layer_call_and_return_conditional_losses_68633126

inputs(
$dense_matmul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identityѕб,batch_normalization/batchnorm/ReadVariableOpб.batch_normalization/batchnorm/ReadVariableOp_1б.batch_normalization/batchnorm/ReadVariableOp_2б0batch_normalization/batchnorm/mul/ReadVariableOpб5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1б5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1б0conv2d_transpose/conv2d_transpose/ReadVariableOpб2conv2d_transpose_1/conv2d_transpose/ReadVariableOpб)conv2d_transpose_2/BiasAdd/ReadVariableOpб2conv2d_transpose_2/conv2d_transpose/ReadVariableOpбdense/MatMul/ReadVariableOpА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђb*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђb2
dense/MatMul¤
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:ђb*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/y┘
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђb2#
!batch_normalization/batchnorm/addа
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ђb2%
#batch_normalization/batchnorm/Rsqrt█
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђb*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpо
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђb2#
!batch_normalization/batchnorm/mul├
#batch_normalization/batchnorm/mul_1Muldense/MatMul:product:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђb2%
#batch_normalization/batchnorm/mul_1Н
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:ђb*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1о
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђb2%
#batch_normalization/batchnorm/mul_2Н
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:ђb*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2н
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђb2#
!batch_normalization/batchnorm/subо
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђb2%
#batch_normalization/batchnorm/add_1ъ
leaky_re_lu/LeakyRelu	LeakyRelu'batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         ђb*
alpha%џЎЎ>2
leaky_re_lu/LeakyReluq
reshape/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeГ
reshape/ReshapeReshape#leaky_re_lu/LeakyRelu:activations:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђ2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shapeќ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackџ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1џ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2╚
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2w
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose/stack/3Э
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackџ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stackъ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1ъ
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2м
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1У
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpх
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transposeи
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_1/ReadVariableOpй
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_1/ReadVariableOp_1Ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3*conv2d_transpose/conv2d_transpose:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3Г
leaky_re_lu_1/LeakyRelu	LeakyRelu*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>2
leaky_re_lu_1/LeakyReluњ
dropout/IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*0
_output_shapes
:         ђ2
dropout/Identity}
conv2d_transpose_1/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shapeџ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackъ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1ъ
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2н
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_1/stack/3ё
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackъ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stackб
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1б
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2я
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1ь
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpй
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0dropout/Identity:output:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transposeХ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1З
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,conv2d_transpose_1/conv2d_transpose:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3г
leaky_re_lu_2/LeakyRelu	LeakyRelu*batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>2
leaky_re_lu_2/LeakyReluЋ
dropout_1/IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2
dropout_1/Identity
conv2d_transpose_2/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shapeџ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackъ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1ъ
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2н
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3ё
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackъ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stackб
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1б
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2я
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1В
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp┐
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0dropout_1/Identity:output:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose┼
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpя
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_transpose_2/BiasAddЎ
conv2d_transpose_2/TanhTanh#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:         2
conv2d_transpose_2/Tanhд
IdentityIdentityconv2d_transpose_2/Tanh:y:0-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_11^conv2d_transpose/conv2d_transpose/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
═
f
G__inference_dropout_3_layer_call_and_return_conditional_losses_68631266

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeй
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/yК
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2
dropout/GreaterEqualѕ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout/CastЃ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Р	
Џ
/__inference_sequential_1_layer_call_fn_68633382

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686318052
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:+                           ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ц
Ф
8__inference_batch_normalization_2_layer_call_fn_68633869

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_686302192
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
█
џ
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68633737

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3Г
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
х
c
E__inference_dropout_layer_call_and_return_conditional_losses_68630467

inputs

identity_1u
IdentityIdentityinputs*
T0*B
_output_shapes0
.:,                           ђ2

Identityё

Identity_1IdentityIdentity:output:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ш
ъ
C__inference_dense_layer_call_and_return_conditional_losses_68630317

inputs"
matmul_readvariableop_resource
identityѕбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђb*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђb2
MatMul}
IdentityIdentityMatMul:product:0^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*+
_input_shapes
:         ђ:2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┼
f
G__inference_dropout_2_layer_call_and_return_conditional_losses_68634104

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ш
Ф
/__inference_sequential_2_layer_call_fn_68632856

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *?
_read_only_resource_inputs!
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_686322052
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
љ
L
0__inference_leaky_re_lu_1_layer_call_fn_68633791

inputs
identityу
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_686304422
PartitionedCallЄ
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ЃE
Б
H__inference_sequential_layer_call_and_return_conditional_losses_68630567
dense_input
dense_68630326 
batch_normalization_68630355 
batch_normalization_68630357 
batch_normalization_68630359 
batch_normalization_68630361
conv2d_transpose_68630399"
batch_normalization_1_68630428"
batch_normalization_1_68630430"
batch_normalization_1_68630432"
batch_normalization_1_68630434
conv2d_transpose_1_68630480"
batch_normalization_2_68630509"
batch_normalization_2_68630511"
batch_normalization_2_68630513"
batch_normalization_2_68630515
conv2d_transpose_2_68630561
conv2d_transpose_2_68630563
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallбdense/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallѓ
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_68630326*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_686303172
dense/StatefulPartitionedCall│
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_68630355batch_normalization_68630357batch_normalization_68630359batch_normalization_68630361*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_686299312-
+batch_normalization/StatefulPartitionedCallЉ
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_686303692
leaky_re_lu/PartitionedCall§
reshape/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_686303912
reshape/PartitionedCallП
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_68630399*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_686300062*
(conv2d_transpose/StatefulPartitionedCallТ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_1_68630428batch_normalization_1_68630430batch_normalization_1_68630432batch_normalization_1_68630434*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_686300762/
-batch_normalization_1/StatefulPartitionedCall│
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_686304422
leaky_re_lu_1/PartitionedCallЕ
dropout/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_686304622!
dropout/StatefulPartitionedCallВ
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_transpose_1_68630480*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_686301492,
*conv2d_transpose_1/StatefulPartitionedCallу
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_2_68630509batch_normalization_2_68630511batch_normalization_2_68630513batch_normalization_2_68630515*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_686302192/
-batch_normalization_2/StatefulPartitionedCall▓
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_686305232
leaky_re_lu_2/PartitionedCallл
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_686305432#
!dropout_1/StatefulPartitionedCallЇ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_transpose_2_68630561conv2d_transpose_2_68630563*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_686302962,
*conv2d_transpose_2/StatefulPartitionedCallџ
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:U Q
(
_output_shapes
:         ђ
%
_user_specified_namedense_input
Р
Ф
8__inference_batch_normalization_4_layer_call_fn_68634244

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_686312002
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ф
F
*__inference_reshape_layer_call_fn_68633717

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_686303912
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђb:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
ч
Ш
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68634056

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Х7
ў
J__inference_sequential_1_layer_call_and_return_conditional_losses_68631420

inputs
conv2d_68631385"
batch_normalization_3_68631388"
batch_normalization_3_68631390"
batch_normalization_3_68631392"
batch_normalization_3_68631394
conv2d_1_68631399"
batch_normalization_4_68631402"
batch_normalization_4_68631404"
batch_normalization_4_68631406"
batch_normalization_4_68631408
dense_1_68631414
dense_1_68631416
identityѕб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense_1/StatefulPartitionedCallб!dropout_2/StatefulPartitionedCallб!dropout_3/StatefulPartitionedCallб&gaussian_noise/StatefulPartitionedCallІ
&gaussian_noise/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_686310142(
&gaussian_noise/StatefulPartitionedCall▒
conv2d/StatefulPartitionedCallStatefulPartitionedCall/gaussian_noise/StatefulPartitionedCall:output:0conv2d_68631385*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_686310382 
conv2d/StatefulPartitionedCall╦
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3_68631388batch_normalization_3_68631390batch_normalization_3_68631392batch_normalization_3_68631394*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_686310672/
-batch_normalization_3/StatefulPartitionedCallа
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_686311262
leaky_re_lu_3/PartitionedCall┼
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0'^gaussian_noise/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_686311462#
!dropout_2/StatefulPartitionedCallх
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_1_68631399*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_686311712"
 conv2d_1/StatefulPartitionedCall╬
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_4_68631402batch_normalization_4_68631404batch_normalization_4_68631406batch_normalization_4_68631408*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_686312002/
-batch_normalization_4/StatefulPartitionedCallЛ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_686312662#
!dropout_3/StatefulPartitionedCallЋ
leaky_re_lu_4/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_686312892
leaky_re_lu_4/PartitionedCallэ
flatten/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_686313032
flatten/PartitionedCall▓
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_68631414dense_1_68631416*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_686313222!
dense_1/StatefulPartitionedCall│
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall'^gaussian_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2P
&gaussian_noise/StatefulPartitionedCall&gaussian_noise/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
№2
х
J__inference_sequential_1_layer_call_and_return_conditional_losses_68631378
gaussian_noise_input
conv2d_68631343"
batch_normalization_3_68631346"
batch_normalization_3_68631348"
batch_normalization_3_68631350"
batch_normalization_3_68631352
conv2d_1_68631357"
batch_normalization_4_68631360"
batch_normalization_4_68631362"
batch_normalization_4_68631364"
batch_normalization_4_68631366
dense_1_68631372
dense_1_68631374
identityѕб-batch_normalization_3/StatefulPartitionedCallб-batch_normalization_4/StatefulPartitionedCallбconv2d/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense_1/StatefulPartitionedCallЂ
gaussian_noise/PartitionedCallPartitionedCallgaussian_noise_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_686310182 
gaussian_noise/PartitionedCallЕ
conv2d/StatefulPartitionedCallStatefulPartitionedCall'gaussian_noise/PartitionedCall:output:0conv2d_68631343*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_686310382 
conv2d/StatefulPartitionedCall╦
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3_68631346batch_normalization_3_68631348batch_normalization_3_68631350batch_normalization_3_68631352*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_686310852/
-batch_normalization_3/StatefulPartitionedCallа
leaky_re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_686311262
leaky_re_lu_3/PartitionedCallё
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_686311512
dropout_2/PartitionedCallГ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_1_68631357*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_686311712"
 conv2d_1/StatefulPartitionedCall╬
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_4_68631360batch_normalization_4_68631362batch_normalization_4_68631364batch_normalization_4_68631366*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_686312182/
-batch_normalization_4/StatefulPartitionedCallЋ
dropout_3/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_686312712
dropout_3/PartitionedCallЇ
leaky_re_lu_4/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_686312892
leaky_re_lu_4/PartitionedCallэ
flatten/PartitionedCallPartitionedCall&leaky_re_lu_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_686313032
flatten/PartitionedCall▓
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_68631372dense_1_68631374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_686313222!
dense_1/StatefulPartitionedCall┬
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:e a
/
_output_shapes
:         
.
_user_specified_namegaussian_noise_input
З	
я
E__inference_dense_1_layer_call_and_return_conditional_losses_68631322

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoidљ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ1::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ1
 
_user_specified_nameinputs
Є
Ш
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634231

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¤
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3█
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
д
Ф
8__inference_batch_normalization_3_layer_call_fn_68634020

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_686308882
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Р
ѕ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_68633662

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђb2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђb2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђb2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђb2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђb::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
ч
Ш
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68631067

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
јi
Ѓ

J__inference_sequential_1_layer_call_and_return_conditional_losses_68631805

inputs)
%conv2d_conv2d_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1бconv2d/Conv2D/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpb
gaussian_noise/ShapeShapeinputs*
T0*
_output_shapes
:2
gaussian_noise/ShapeІ
!gaussian_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!gaussian_noise/random_normal/meanЈ
#gaussian_noise/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2%
#gaussian_noise/random_normal/stddevЋ
1gaussian_noise/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise/Shape:output:0*
T0*A
_output_shapes/
-:+                           *
dtype0*
seed▒ т)*
seed2ът■23
1gaussian_noise/random_normal/RandomStandardNormalЂ
 gaussian_noise/random_normal/mulMul:gaussian_noise/random_normal/RandomStandardNormal:output:0,gaussian_noise/random_normal/stddev:output:0*
T0*A
_output_shapes/
-:+                           2"
 gaussian_noise/random_normal/mulр
gaussian_noise/random_normalAdd$gaussian_noise/random_normal/mul:z:0*gaussian_noise/random_normal/mean:output:0*
T0*A
_output_shapes/
-:+                           2
gaussian_noise/random_normalД
gaussian_noise/addAddV2inputs gaussian_noise/random_normal:z:0*
T0*A
_output_shapes/
-:+                           2
gaussian_noise/addф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp┌
conv2d/Conv2DConv2Dgaussian_noise/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
conv2d/Conv2DХ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1­
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Й
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*A
_output_shapes/
-:+                           @*
alpha%џЎЎ>2
leaky_re_lu_3/LeakyReluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout_2/dropout/Const╩
dropout_2/dropout/MulMul%leaky_re_lu_3/LeakyRelu:activations:0 dropout_2/dropout/Const:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout_2/dropout/MulЄ
dropout_2/dropout/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeВ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЅ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2"
 dropout_2/dropout/GreaterEqual/yђ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*A
_output_shapes/
-:+                           @2 
dropout_2/dropout/GreaterEqualи
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout_2/dropout/Cast╝
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout_2/dropout/Mul_1▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpТ
conv2d_1/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
2
conv2d_1/Conv2Dи
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_4/ReadVariableOpй
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_4/ReadVariableOp_1Ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout_3/dropout/Constл
dropout_3/dropout/MulMul*batch_normalization_4/FusedBatchNormV3:y:0 dropout_3/dropout/Const:output:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout_3/dropout/Mulї
dropout_3/dropout/ShapeShape*batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeь
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*B
_output_shapes0
.:,                           ђ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЅ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2"
 dropout_3/dropout/GreaterEqual/yЂ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*B
_output_shapes0
.:,                           ђ2 
dropout_3/dropout/GreaterEqualИ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*B
_output_shapes0
.:,                           ђ2
dropout_3/dropout/Castй
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout_3/dropout/Mul_1░
leaky_re_lu_4/LeakyRelu	LeakyReludropout_3/dropout/Mul_1:z:0*B
_output_shapes0
.:,                           ђ*
alpha%џЎЎ>2
leaky_re_lu_4/LeakyRelus
flatten/ShapeShape%leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
flatten/Shapeё
flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
flatten/strided_slice/stackѕ
flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
flatten/strided_slice/stack_1ѕ
flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
flatten/strided_slice/stack_2њ
flatten/strided_sliceStridedSliceflatten/Shape:output:0$flatten/strided_slice/stack:output:0&flatten/strided_slice/stack_1:output:0&flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
flatten/strided_slice}
flatten/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2
flatten/Reshape/shape/1д
flatten/Reshape/shapePackflatten/strided_slice:output:0 flatten/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
flatten/Reshape/shape»
flatten/ReshapeReshape%leaky_re_lu_4/LeakyRelu:activations:0flatten/Reshape/shape:output:0*
T0*0
_output_shapes
:                  2
flatten/Reshapeд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/SigmoidВ
IdentityIdentitydense_1/Sigmoid:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:+                           ::::::::::::2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Њ
Ь
-__inference_sequential_layer_call_fn_68630709
dense_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identityѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *-
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_686306722
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:         ђ
%
_user_specified_namedense_input
┌$
§
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_68630296

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3ѓ
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2В
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3│
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOp­
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
2
conv2d_transposeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpц
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Tanh▒
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ў
Ь
-__inference_sequential_layer_call_fn_68630799
dense_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *3
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_686307622
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:         ђ
%
_user_specified_namedense_input
¤
Ш
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68633755

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Р
ѕ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_68629964

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpЊ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЅ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђb2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђb2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђb2
batchnorm/mul_1Ў
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/ReadVariableOp_1є
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/mul_2Ў
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/ReadVariableOp_2ё
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђb2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђb::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
Ј
c
*__inference_dropout_layer_call_fn_68633813

inputs
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_686304622
StatefulPartitionedCallЕ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
љщ
Ю
H__inference_sequential_layer_call_and_return_conditional_losses_68633008

inputs(
$dense_matmul_readvariableop_resource0
,batch_normalization_assignmovingavg_686328702
.batch_normalization_assignmovingavg_1_68632876=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identityѕб7batch_normalization/AssignMovingAvg/AssignSubVariableOpб2batch_normalization/AssignMovingAvg/ReadVariableOpб9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpб4batch_normalization/AssignMovingAvg_1/ReadVariableOpб,batch_normalization/batchnorm/ReadVariableOpб0batch_normalization/batchnorm/mul/ReadVariableOpб$batch_normalization_1/AssignNewValueб&batch_normalization_1/AssignNewValue_1б5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1б$batch_normalization_2/AssignNewValueб&batch_normalization_2/AssignNewValue_1б5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1б0conv2d_transpose/conv2d_transpose/ReadVariableOpб2conv2d_transpose_1/conv2d_transpose/ReadVariableOpб)conv2d_transpose_2/BiasAdd/ReadVariableOpб2conv2d_transpose_2/conv2d_transpose/ReadVariableOpбdense/MatMul/ReadVariableOpА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ђђb*
dtype02
dense/MatMul/ReadVariableOpє
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђb2
dense/MatMul▓
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indices▄
 batch_normalization/moments/meanMeandense/MatMul:product:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђb*
	keep_dims(2"
 batch_normalization/moments/mean╣
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	ђb2*
(batch_normalization/moments/StopGradientы
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense/MatMul:product:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђb2/
-batch_normalization/moments/SquaredDifference║
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indicesЃ
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђb*
	keep_dims(2&
$batch_normalization/moments/varianceй
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:ђb*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:ђb*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1і
)batch_normalization/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization/AssignMovingAvg/68632870*
_output_shapes
: *
dtype0*
valueB
 *
О#<2+
)batch_normalization/AssignMovingAvg/decayм
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_assignmovingavg_68632870*
_output_shapes	
:ђb*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpп
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg/68632870*
_output_shapes	
:ђb2)
'batch_normalization/AssignMovingAvg/sub¤
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg/68632870*
_output_shapes	
:ђb2)
'batch_normalization/AssignMovingAvg/mulФ
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_assignmovingavg_68632870+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization/AssignMovingAvg/68632870*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpљ
+batch_normalization/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization/AssignMovingAvg_1/68632876*
_output_shapes
: *
dtype0*
valueB
 *
О#<2-
+batch_normalization/AssignMovingAvg_1/decayп
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_assignmovingavg_1_68632876*
_output_shapes	
:ђb*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpР
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization/AssignMovingAvg_1/68632876*
_output_shapes	
:ђb2+
)batch_normalization/AssignMovingAvg_1/sub┘
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization/AssignMovingAvg_1/68632876*
_output_shapes	
:ђb2+
)batch_normalization/AssignMovingAvg_1/mulи
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_assignmovingavg_1_68632876-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization/AssignMovingAvg_1/68632876*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpЈ
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2%
#batch_normalization/batchnorm/add/yМ
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђb2#
!batch_normalization/batchnorm/addа
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:ђb2%
#batch_normalization/batchnorm/Rsqrt█
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђb*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpо
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђb2#
!batch_normalization/batchnorm/mul├
#batch_normalization/batchnorm/mul_1Muldense/MatMul:product:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђb2%
#batch_normalization/batchnorm/mul_1╠
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:ђb2%
#batch_normalization/batchnorm/mul_2¤
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:ђb*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpм
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђb2#
!batch_normalization/batchnorm/subо
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђb2%
#batch_normalization/batchnorm/add_1ъ
leaky_re_lu/LeakyRelu	LeakyRelu'batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         ђb*
alpha%џЎЎ>2
leaky_re_lu/LeakyReluq
reshape/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeГ
reshape/ReshapeReshape#leaky_re_lu/LeakyRelu:activations:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђ2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shapeќ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackџ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1џ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2╚
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2w
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
conv2d_transpose/stack/3Э
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackџ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stackъ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1ъ
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2м
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1У
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:ђђ*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpх
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transposeи
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_1/ReadVariableOpй
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_1/ReadVariableOp_1Ж
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ё
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3*conv2d_transpose/conv2d_transpose:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_1/FusedBatchNormV3▒
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue┐
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1Г
leaky_re_lu_1/LeakyRelu	LeakyRelu*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>2
leaky_re_lu_1/LeakyRelus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/dropout/Const│
dropout/dropout/MulMul%leaky_re_lu_1/LeakyRelu:activations:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:         ђ2
dropout/dropout/MulЃ
dropout/dropout/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeН
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:         ђ*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2 
dropout/dropout/GreaterEqual/yу
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         ђ2
dropout/dropout/GreaterEqualа
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         ђ2
dropout/dropout/CastБ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђ2
dropout/dropout/Mul_1}
conv2d_transpose_1/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shapeџ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackъ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1ъ
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2н
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_1/stack/3ё
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackъ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stackб
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1б
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2я
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1ь
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@ђ*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpй
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0dropout/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transposeХ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1ж
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ѓ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3,conv2d_transpose_1/conv2d_transpose:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2(
&batch_normalization_2/FusedBatchNormV3▒
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue┐
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1г
leaky_re_lu_2/LeakyRelu	LeakyRelu*batch_normalization_2/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>2
leaky_re_lu_2/LeakyReluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout_1/dropout/ConstИ
dropout_1/dropout/MulMul%leaky_re_lu_2/LeakyRelu:activations:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout_1/dropout/MulЄ
dropout_1/dropout/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape┌
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЅ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2"
 dropout_1/dropout/GreaterEqual/yЬ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2 
dropout_1/dropout/GreaterEqualЦ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout_1/dropout/Castф
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout_1/dropout/Mul_1
conv2d_transpose_2/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shapeџ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackъ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1ъ
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2н
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3ё
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackъ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stackб
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1б
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2я
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1В
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp┐
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0dropout_1/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose┼
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpя
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_transpose_2/BiasAddЎ
conv2d_transpose_2/TanhTanh#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:         2
conv2d_transpose_2/Tanhк	
IdentityIdentityconv2d_transpose_2/Tanh:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_11^conv2d_transpose/conv2d_transpose/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
├
Ш
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68630250

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpГ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ё
ж
-__inference_sequential_layer_call_fn_68633165

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identityѕбStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *-
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_686306722
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╦Ь
ј2
$__inference__traced_restore_68634880
file_prefix
assignvariableop_yogi_iter"
assignvariableop_1_yogi_beta_1"
assignvariableop_2_yogi_beta_2!
assignvariableop_3_yogi_decay#
assignvariableop_4_yogi_epsilon6
2assignvariableop_5_yogi_l1_regularization_strength6
2assignvariableop_6_yogi_l2_regularization_strength)
%assignvariableop_7_yogi_learning_rate#
assignvariableop_8_dense_kernel0
,assignvariableop_9_batch_normalization_gamma0
,assignvariableop_10_batch_normalization_beta7
3assignvariableop_11_batch_normalization_moving_mean;
7assignvariableop_12_batch_normalization_moving_variance/
+assignvariableop_13_conv2d_transpose_kernel3
/assignvariableop_14_batch_normalization_1_gamma2
.assignvariableop_15_batch_normalization_1_beta9
5assignvariableop_16_batch_normalization_1_moving_mean=
9assignvariableop_17_batch_normalization_1_moving_variance1
-assignvariableop_18_conv2d_transpose_1_kernel3
/assignvariableop_19_batch_normalization_2_gamma2
.assignvariableop_20_batch_normalization_2_beta9
5assignvariableop_21_batch_normalization_2_moving_mean=
9assignvariableop_22_batch_normalization_2_moving_variance1
-assignvariableop_23_conv2d_transpose_2_kernel/
+assignvariableop_24_conv2d_transpose_2_bias%
!assignvariableop_25_conv2d_kernel3
/assignvariableop_26_batch_normalization_3_gamma2
.assignvariableop_27_batch_normalization_3_beta9
5assignvariableop_28_batch_normalization_3_moving_mean=
9assignvariableop_29_batch_normalization_3_moving_variance'
#assignvariableop_30_conv2d_1_kernel3
/assignvariableop_31_batch_normalization_4_gamma2
.assignvariableop_32_batch_normalization_4_beta9
5assignvariableop_33_batch_normalization_4_moving_mean=
9assignvariableop_34_batch_normalization_4_moving_variance&
"assignvariableop_35_dense_1_kernel$
 assignvariableop_36_dense_1_bias#
assignvariableop_37_yogi_iter_1%
!assignvariableop_38_yogi_beta_1_1%
!assignvariableop_39_yogi_beta_2_1$
 assignvariableop_40_yogi_decay_1&
"assignvariableop_41_yogi_epsilon_19
5assignvariableop_42_yogi_l1_regularization_strength_19
5assignvariableop_43_yogi_l2_regularization_strength_1,
(assignvariableop_44_yogi_learning_rate_1
assignvariableop_45_total
assignvariableop_46_count
assignvariableop_47_total_1
assignvariableop_48_count_1+
'assignvariableop_49_yogi_dense_kernel_v8
4assignvariableop_50_yogi_batch_normalization_gamma_v7
3assignvariableop_51_yogi_batch_normalization_beta_v6
2assignvariableop_52_yogi_conv2d_transpose_kernel_v:
6assignvariableop_53_yogi_batch_normalization_1_gamma_v9
5assignvariableop_54_yogi_batch_normalization_1_beta_v8
4assignvariableop_55_yogi_conv2d_transpose_1_kernel_v:
6assignvariableop_56_yogi_batch_normalization_2_gamma_v9
5assignvariableop_57_yogi_batch_normalization_2_beta_v8
4assignvariableop_58_yogi_conv2d_transpose_2_kernel_v6
2assignvariableop_59_yogi_conv2d_transpose_2_bias_v+
'assignvariableop_60_yogi_dense_kernel_m8
4assignvariableop_61_yogi_batch_normalization_gamma_m7
3assignvariableop_62_yogi_batch_normalization_beta_m6
2assignvariableop_63_yogi_conv2d_transpose_kernel_m:
6assignvariableop_64_yogi_batch_normalization_1_gamma_m9
5assignvariableop_65_yogi_batch_normalization_1_beta_m8
4assignvariableop_66_yogi_conv2d_transpose_1_kernel_m:
6assignvariableop_67_yogi_batch_normalization_2_gamma_m9
5assignvariableop_68_yogi_batch_normalization_2_beta_m8
4assignvariableop_69_yogi_conv2d_transpose_2_kernel_m6
2assignvariableop_70_yogi_conv2d_transpose_2_bias_m,
(assignvariableop_71_yogi_conv2d_kernel_v:
6assignvariableop_72_yogi_batch_normalization_3_gamma_v9
5assignvariableop_73_yogi_batch_normalization_3_beta_v.
*assignvariableop_74_yogi_conv2d_1_kernel_v:
6assignvariableop_75_yogi_batch_normalization_4_gamma_v9
5assignvariableop_76_yogi_batch_normalization_4_beta_v-
)assignvariableop_77_yogi_dense_1_kernel_v+
'assignvariableop_78_yogi_dense_1_bias_v,
(assignvariableop_79_yogi_conv2d_kernel_m:
6assignvariableop_80_yogi_batch_normalization_3_gamma_m9
5assignvariableop_81_yogi_batch_normalization_3_beta_m.
*assignvariableop_82_yogi_conv2d_1_kernel_m:
6assignvariableop_83_yogi_batch_normalization_4_gamma_m9
5assignvariableop_84_yogi_batch_normalization_4_beta_m-
)assignvariableop_85_yogi_dense_1_kernel_m+
'assignvariableop_86_yogi_dense_1_bias_m
identity_88ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_75бAssignVariableOp_76бAssignVariableOp_77бAssignVariableOp_78бAssignVariableOp_79бAssignVariableOp_8бAssignVariableOp_80бAssignVariableOp_81бAssignVariableOp_82бAssignVariableOp_83бAssignVariableOp_84бAssignVariableOp_85бAssignVariableOp_86бAssignVariableOp_9ў+
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*ц*
valueџ*BЌ*XB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB,optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUEB?optimizer/l1_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEB?optimizer/l2_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/optimizer/epsilon/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/optimizer/l1_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/optimizer/l2_regularization_strength/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/18/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/19/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/28/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBXvariables/17/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/18/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/19/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/22/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/23/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/24/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/27/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBXvariables/28/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names┴
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*┼
value╗BИXB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesТ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ш
_output_shapesс
Я::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*f
dtypes\
Z2X		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

IdentityЎ
AssignVariableOpAssignVariableOpassignvariableop_yogi_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Б
AssignVariableOp_1AssignVariableOpassignvariableop_1_yogi_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Б
AssignVariableOp_2AssignVariableOpassignvariableop_2_yogi_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3б
AssignVariableOp_3AssignVariableOpassignvariableop_3_yogi_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ц
AssignVariableOp_4AssignVariableOpassignvariableop_4_yogi_epsilonIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5и
AssignVariableOp_5AssignVariableOp2assignvariableop_5_yogi_l1_regularization_strengthIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6и
AssignVariableOp_6AssignVariableOp2assignvariableop_6_yogi_l2_regularization_strengthIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ф
AssignVariableOp_7AssignVariableOp%assignvariableop_7_yogi_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ц
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9▒
AssignVariableOp_9AssignVariableOp,assignvariableop_9_batch_normalization_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┤
AssignVariableOp_10AssignVariableOp,assignvariableop_10_batch_normalization_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╗
AssignVariableOp_11AssignVariableOp3assignvariableop_11_batch_normalization_moving_meanIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12┐
AssignVariableOp_12AssignVariableOp7assignvariableop_12_batch_normalization_moving_varianceIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13│
AssignVariableOp_13AssignVariableOp+assignvariableop_13_conv2d_transpose_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14и
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_1_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Х
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_1_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16й
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_1_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17┴
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_1_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18х
AssignVariableOp_18AssignVariableOp-assignvariableop_18_conv2d_transpose_1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19и
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_2_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Х
AssignVariableOp_20AssignVariableOp.assignvariableop_20_batch_normalization_2_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21й
AssignVariableOp_21AssignVariableOp5assignvariableop_21_batch_normalization_2_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22┴
AssignVariableOp_22AssignVariableOp9assignvariableop_22_batch_normalization_2_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23х
AssignVariableOp_23AssignVariableOp-assignvariableop_23_conv2d_transpose_2_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24│
AssignVariableOp_24AssignVariableOp+assignvariableop_24_conv2d_transpose_2_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Е
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26и
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_3_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Х
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_3_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28й
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_3_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29┴
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_3_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ф
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31и
AssignVariableOp_31AssignVariableOp/assignvariableop_31_batch_normalization_4_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Х
AssignVariableOp_32AssignVariableOp.assignvariableop_32_batch_normalization_4_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33й
AssignVariableOp_33AssignVariableOp5assignvariableop_33_batch_normalization_4_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34┴
AssignVariableOp_34AssignVariableOp9assignvariableop_34_batch_normalization_4_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ф
AssignVariableOp_35AssignVariableOp"assignvariableop_35_dense_1_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36е
AssignVariableOp_36AssignVariableOp assignvariableop_36_dense_1_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_37Д
AssignVariableOp_37AssignVariableOpassignvariableop_37_yogi_iter_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Е
AssignVariableOp_38AssignVariableOp!assignvariableop_38_yogi_beta_1_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Е
AssignVariableOp_39AssignVariableOp!assignvariableop_39_yogi_beta_2_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40е
AssignVariableOp_40AssignVariableOp assignvariableop_40_yogi_decay_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41ф
AssignVariableOp_41AssignVariableOp"assignvariableop_41_yogi_epsilon_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42й
AssignVariableOp_42AssignVariableOp5assignvariableop_42_yogi_l1_regularization_strength_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43й
AssignVariableOp_43AssignVariableOp5assignvariableop_43_yogi_l2_regularization_strength_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44░
AssignVariableOp_44AssignVariableOp(assignvariableop_44_yogi_learning_rate_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45А
AssignVariableOp_45AssignVariableOpassignvariableop_45_totalIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46А
AssignVariableOp_46AssignVariableOpassignvariableop_46_countIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Б
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_1Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Б
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_1Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49»
AssignVariableOp_49AssignVariableOp'assignvariableop_49_yogi_dense_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50╝
AssignVariableOp_50AssignVariableOp4assignvariableop_50_yogi_batch_normalization_gamma_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51╗
AssignVariableOp_51AssignVariableOp3assignvariableop_51_yogi_batch_normalization_beta_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52║
AssignVariableOp_52AssignVariableOp2assignvariableop_52_yogi_conv2d_transpose_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Й
AssignVariableOp_53AssignVariableOp6assignvariableop_53_yogi_batch_normalization_1_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54й
AssignVariableOp_54AssignVariableOp5assignvariableop_54_yogi_batch_normalization_1_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55╝
AssignVariableOp_55AssignVariableOp4assignvariableop_55_yogi_conv2d_transpose_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Й
AssignVariableOp_56AssignVariableOp6assignvariableop_56_yogi_batch_normalization_2_gamma_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57й
AssignVariableOp_57AssignVariableOp5assignvariableop_57_yogi_batch_normalization_2_beta_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58╝
AssignVariableOp_58AssignVariableOp4assignvariableop_58_yogi_conv2d_transpose_2_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59║
AssignVariableOp_59AssignVariableOp2assignvariableop_59_yogi_conv2d_transpose_2_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60»
AssignVariableOp_60AssignVariableOp'assignvariableop_60_yogi_dense_kernel_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61╝
AssignVariableOp_61AssignVariableOp4assignvariableop_61_yogi_batch_normalization_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62╗
AssignVariableOp_62AssignVariableOp3assignvariableop_62_yogi_batch_normalization_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63║
AssignVariableOp_63AssignVariableOp2assignvariableop_63_yogi_conv2d_transpose_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Й
AssignVariableOp_64AssignVariableOp6assignvariableop_64_yogi_batch_normalization_1_gamma_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65й
AssignVariableOp_65AssignVariableOp5assignvariableop_65_yogi_batch_normalization_1_beta_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66╝
AssignVariableOp_66AssignVariableOp4assignvariableop_66_yogi_conv2d_transpose_1_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Й
AssignVariableOp_67AssignVariableOp6assignvariableop_67_yogi_batch_normalization_2_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68й
AssignVariableOp_68AssignVariableOp5assignvariableop_68_yogi_batch_normalization_2_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69╝
AssignVariableOp_69AssignVariableOp4assignvariableop_69_yogi_conv2d_transpose_2_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70║
AssignVariableOp_70AssignVariableOp2assignvariableop_70_yogi_conv2d_transpose_2_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71░
AssignVariableOp_71AssignVariableOp(assignvariableop_71_yogi_conv2d_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Й
AssignVariableOp_72AssignVariableOp6assignvariableop_72_yogi_batch_normalization_3_gamma_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73й
AssignVariableOp_73AssignVariableOp5assignvariableop_73_yogi_batch_normalization_3_beta_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74▓
AssignVariableOp_74AssignVariableOp*assignvariableop_74_yogi_conv2d_1_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Й
AssignVariableOp_75AssignVariableOp6assignvariableop_75_yogi_batch_normalization_4_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76й
AssignVariableOp_76AssignVariableOp5assignvariableop_76_yogi_batch_normalization_4_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77▒
AssignVariableOp_77AssignVariableOp)assignvariableop_77_yogi_dense_1_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78»
AssignVariableOp_78AssignVariableOp'assignvariableop_78_yogi_dense_1_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79░
AssignVariableOp_79AssignVariableOp(assignvariableop_79_yogi_conv2d_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80Й
AssignVariableOp_80AssignVariableOp6assignvariableop_80_yogi_batch_normalization_3_gamma_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81й
AssignVariableOp_81AssignVariableOp5assignvariableop_81_yogi_batch_normalization_3_beta_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82▓
AssignVariableOp_82AssignVariableOp*assignvariableop_82_yogi_conv2d_1_kernel_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83Й
AssignVariableOp_83AssignVariableOp6assignvariableop_83_yogi_batch_normalization_4_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84й
AssignVariableOp_84AssignVariableOp5assignvariableop_84_yogi_batch_normalization_4_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85▒
AssignVariableOp_85AssignVariableOp)assignvariableop_85_yogi_dense_1_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86»
AssignVariableOp_86AssignVariableOp'assignvariableop_86_yogi_dense_1_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_869
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpп
Identity_87Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_87╦
Identity_88IdentityIdentity_87:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_88"#
identity_88Identity_88:output:0*з
_input_shapesр
я: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
█
џ
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68630076

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<2
FusedBatchNormV3Г
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Љ
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Э
g
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_68631289

inputs
identitym
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:         ђ*
alpha%џЎЎ>2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
в
a
E__inference_reshape_layer_call_and_return_conditional_losses_68630391

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :ђ2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:         ђ2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђb:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
в@
Ѓ

J__inference_sequential_1_layer_call_and_return_conditional_losses_68633534

inputs)
%conv2d_conv2d_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1бconv2d/Conv2D/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpИ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d/Conv2DХ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1я
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3г
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
alpha%џЎЎ>2
leaky_re_lu_3/LeakyReluЋ
dropout_2/IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*/
_output_shapes
:         @2
dropout_2/Identity▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpн
conv2d_1/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_1/Conv2Dи
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_4/ReadVariableOpй
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_4/ReadVariableOp_1Ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1т
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3Џ
dropout_3/IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:         ђ2
dropout_3/Identityъ
leaky_re_lu_4/LeakyRelu	LeakyReludropout_3/Identity:output:0*0
_output_shapes
:         ђ*
alpha%џЎЎ>2
leaky_re_lu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  2
flatten/ConstЪ
flatten/ReshapeReshape%leaky_re_lu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ12
flatten/Reshapeд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/SigmoidВ
IdentityIdentitydense_1/Sigmoid:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┴
n
(__inference_dense_layer_call_fn_68633606

inputs
unknown
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_686303172
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*+
_input_shapes
:         ђ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ж
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_68631151

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
У	
Е
/__inference_sequential_1_layer_call_fn_68631447
gaussian_noise_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallgaussian_noise_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686314202
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:         
.
_user_specified_namegaussian_noise_input
Ь
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_68631271

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:         ђ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Њ
х
/__inference_sequential_2_layer_call_fn_68632266
sequential_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *?
_read_only_resource_inputs!
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_686322052
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Ю
_input_shapesІ
ѕ:         ђ:::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:         ђ
*
_user_specified_namesequential_input
З
g
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_68631126

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         @*
alpha%џЎЎ>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
│
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_68633909

inputs

identity_1t
IdentityIdentityinputs*
T0*A
_output_shapes/
-:+                           @2

IdentityЃ

Identity_1IdentityIdentity:output:0*
T0*A
_output_shapes/
-:+                           @2

Identity_1"!

identity_1Identity_1:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
¤
Ш
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634151

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype02
ReadVariableOp_1е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
FusedBatchNormV3/ReadVariableOp«
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1р
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Й	
Џ
/__inference_sequential_1_layer_call_fn_68633592

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686314882
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:         ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
д
Ф
8__inference_batch_normalization_2_layer_call_fn_68633882

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_686302502
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
К
L
0__inference_leaky_re_lu_4_layer_call_fn_68634294

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_686312892
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Э
g
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_68634289

inputs
identitym
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:         ђ*
alpha%џЎЎ>2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
о
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_68633693

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:         ђb*
alpha%џЎЎ>2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђb:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
ђB
п
H__inference_sequential_layer_call_and_return_conditional_losses_68630762

inputs
dense_68630714 
batch_normalization_68630717 
batch_normalization_68630719 
batch_normalization_68630721 
batch_normalization_68630723
conv2d_transpose_68630728"
batch_normalization_1_68630731"
batch_normalization_1_68630733"
batch_normalization_1_68630735"
batch_normalization_1_68630737
conv2d_transpose_1_68630742"
batch_normalization_2_68630745"
batch_normalization_2_68630747"
batch_normalization_2_68630749"
batch_normalization_2_68630751
conv2d_transpose_2_68630756
conv2d_transpose_2_68630758
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallбdense/StatefulPartitionedCall§
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_68630714*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_686303172
dense/StatefulPartitionedCallх
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_68630717batch_normalization_68630719batch_normalization_68630721batch_normalization_68630723*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_686299642-
+batch_normalization/StatefulPartitionedCallЉ
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_686303692
leaky_re_lu/PartitionedCall§
reshape/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_686303912
reshape/PartitionedCallП
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_68630728*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_686300062*
(conv2d_transpose/StatefulPartitionedCallУ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_1_68630731batch_normalization_1_68630733batch_normalization_1_68630735batch_normalization_1_68630737*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_686301072/
-batch_normalization_1/StatefulPartitionedCall│
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_686304422
leaky_re_lu_1/PartitionedCallЉ
dropout/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_686304672
dropout/PartitionedCallС
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_transpose_1_68630742*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_686301492,
*conv2d_transpose_1/StatefulPartitionedCallж
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_2_68630745batch_normalization_2_68630747batch_normalization_2_68630749batch_normalization_2_68630751*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_686302502/
-batch_normalization_2/StatefulPartitionedCall▓
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_686305232
leaky_re_lu_2/PartitionedCallќ
dropout_1/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_686305482
dropout_1/PartitionedCallЁ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_transpose_2_68630756conv2d_transpose_2_68630758*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_686302962,
*conv2d_transpose_2/StatefulPartitionedCallн
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
о
f
G__inference_dropout_1_layer_call_and_return_conditional_losses_68630543

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/ConstЇ
dropout/MulMulinputsdropout/Const:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/yп
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout/GreaterEqualЎ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout/Castћ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
П
d
E__inference_dropout_layer_call_and_return_conditional_losses_68633803

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout/Constј
dropout/MulMulinputsdropout/Const:output:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*B
_output_shapes0
.:,                           ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2
dropout/GreaterEqual/y┘
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout/GreaterEqualџ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*B
_output_shapes0
.:,                           ђ2
dropout/CastЋ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout/Mul_1ђ
IdentityIdentitydropout/Mul_1:z:0*
T0*B
_output_shapes0
.:,                           ђ2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,                           ђ:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
░

k
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_68633930

inputs
identityѕD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
random_normal/stddevН
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:         *
dtype0*
seed▒ т)*
seed2ўЎ%2$
"random_normal/RandomStandardNormal│
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:         2
random_normal/mulЊ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:         2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:         2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Я
і
5__inference_conv2d_transpose_2_layer_call_fn_68630306

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_686302962
StatefulPartitionedCallе
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
й
g
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_68633887

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+                           @*
alpha%џЎЎ>2
	LeakyReluЁ
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+                           @:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╗0
╠
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_68629931

inputs
assignmovingavg_68629906
assignmovingavg_1_68629912)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђb*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђb2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђb2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђb*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђb*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђb*
squeeze_dims
 2
moments/Squeeze_1╬
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg/68629906*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayќ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_68629906*
_output_shapes	
:ђb*
dtype02 
AssignMovingAvg/ReadVariableOpЗ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg/68629906*
_output_shapes	
:ђb2
AssignMovingAvg/subв
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg/68629906*
_output_shapes	
:ђb2
AssignMovingAvg/mul│
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_68629906AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg/68629906*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpн
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*-
_class#
!loc:@AssignMovingAvg_1/68629912*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayю
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_68629912*
_output_shapes	
:ђb*
dtype02"
 AssignMovingAvg_1/ReadVariableOp■
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/68629912*
_output_shapes	
:ђb2
AssignMovingAvg_1/subш
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/68629912*
_output_shapes	
:ђb2
AssignMovingAvg_1/mul┐
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_68629912AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*-
_class#
!loc:@AssignMovingAvg_1/68629912*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђb2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђb2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђb2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђb2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђb::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs
јi
Ѓ

J__inference_sequential_1_layer_call_and_return_conditional_losses_68633296

inputs)
%conv2d_conv2d_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1бconv2d/Conv2D/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpb
gaussian_noise/ShapeShapeinputs*
T0*
_output_shapes
:2
gaussian_noise/ShapeІ
!gaussian_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!gaussian_noise/random_normal/meanЈ
#gaussian_noise/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2%
#gaussian_noise/random_normal/stddevЋ
1gaussian_noise/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise/Shape:output:0*
T0*A
_output_shapes/
-:+                           *
dtype0*
seed▒ т)*
seed2▀иЗ23
1gaussian_noise/random_normal/RandomStandardNormalЂ
 gaussian_noise/random_normal/mulMul:gaussian_noise/random_normal/RandomStandardNormal:output:0,gaussian_noise/random_normal/stddev:output:0*
T0*A
_output_shapes/
-:+                           2"
 gaussian_noise/random_normal/mulр
gaussian_noise/random_normalAdd$gaussian_noise/random_normal/mul:z:0*gaussian_noise/random_normal/mean:output:0*
T0*A
_output_shapes/
-:+                           2
gaussian_noise/random_normalД
gaussian_noise/addAddV2inputs gaussian_noise/random_normal:z:0*
T0*A
_output_shapes/
-:+                           2
gaussian_noise/addф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp┌
conv2d/Conv2DConv2Dgaussian_noise/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
conv2d/Conv2DХ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1­
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Й
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*A
_output_shapes/
-:+                           @*
alpha%џЎЎ>2
leaky_re_lu_3/LeakyReluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout_2/dropout/Const╩
dropout_2/dropout/MulMul%leaky_re_lu_3/LeakyRelu:activations:0 dropout_2/dropout/Const:output:0*
T0*A
_output_shapes/
-:+                           @2
dropout_2/dropout/MulЄ
dropout_2/dropout/ShapeShape%leaky_re_lu_3/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeВ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*A
_output_shapes/
-:+                           @*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЅ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2"
 dropout_2/dropout/GreaterEqual/yђ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*A
_output_shapes/
-:+                           @2 
dropout_2/dropout/GreaterEqualи
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*A
_output_shapes/
-:+                           @2
dropout_2/dropout/Cast╝
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*A
_output_shapes/
-:+                           @2
dropout_2/dropout/Mul_1▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpТ
conv2d_1/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
2
conv2d_1/Conv2Dи
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_4/ReadVariableOpй
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_4/ReadVariableOp_1Ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?2
dropout_3/dropout/Constл
dropout_3/dropout/MulMul*batch_normalization_4/FusedBatchNormV3:y:0 dropout_3/dropout/Const:output:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout_3/dropout/Mulї
dropout_3/dropout/ShapeShape*batch_normalization_4/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeь
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*B
_output_shapes0
.:,                           ђ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЅ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>2"
 dropout_3/dropout/GreaterEqual/yЂ
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*B
_output_shapes0
.:,                           ђ2 
dropout_3/dropout/GreaterEqualИ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*B
_output_shapes0
.:,                           ђ2
dropout_3/dropout/Castй
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout_3/dropout/Mul_1░
leaky_re_lu_4/LeakyRelu	LeakyReludropout_3/dropout/Mul_1:z:0*B
_output_shapes0
.:,                           ђ*
alpha%џЎЎ>2
leaky_re_lu_4/LeakyRelus
flatten/ShapeShape%leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
flatten/Shapeё
flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
flatten/strided_slice/stackѕ
flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
flatten/strided_slice/stack_1ѕ
flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
flatten/strided_slice/stack_2њ
flatten/strided_sliceStridedSliceflatten/Shape:output:0$flatten/strided_slice/stack:output:0&flatten/strided_slice/stack_1:output:0&flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
flatten/strided_slice}
flatten/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2
flatten/Reshape/shape/1д
flatten/Reshape/shapePackflatten/strided_slice:output:0 flatten/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
flatten/Reshape/shape»
flatten/ReshapeReshape%leaky_re_lu_4/LeakyRelu:activations:0flatten/Reshape/shape:output:0*
T0*0
_output_shapes
:                  2
flatten/Reshapeд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/SigmoidВ
IdentityIdentitydense_1/Sigmoid:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:+                           ::::::::::::2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┬
Ъ
D__inference_conv2d_layer_call_and_return_conditional_losses_68631038

inputs"
conv2d_readvariableop_resource
identityѕбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
Conv2DЃ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
▀
o
)__inference_conv2d_layer_call_fn_68633958

inputs
unknown
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_686310382
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ЇJ
Ѓ

J__inference_sequential_1_layer_call_and_return_conditional_losses_68633353

inputs)
%conv2d_conv2d_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1б5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1бconv2d/Conv2D/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpф
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╩
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
2
conv2d/Conv2DХ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp╝
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1ж
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp№
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1­
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Й
leaky_re_lu_3/LeakyRelu	LeakyRelu*batch_normalization_3/FusedBatchNormV3:y:0*A
_output_shapes/
-:+                           @*
alpha%џЎЎ>2
leaky_re_lu_3/LeakyReluД
dropout_2/IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2
dropout_2/Identity▒
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02 
conv2d_1/Conv2D/ReadVariableOpТ
conv2d_1/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
2
conv2d_1/Conv2Dи
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ђ*
dtype02&
$batch_normalization_4/ReadVariableOpй
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype02(
&batch_normalization_4/ReadVariableOp_1Ж
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp­
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1э
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3Г
dropout_3/IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,                           ђ2
dropout_3/Identity░
leaky_re_lu_4/LeakyRelu	LeakyReludropout_3/Identity:output:0*B
_output_shapes0
.:,                           ђ*
alpha%џЎЎ>2
leaky_re_lu_4/LeakyRelus
flatten/ShapeShape%leaky_re_lu_4/LeakyRelu:activations:0*
T0*
_output_shapes
:2
flatten/Shapeё
flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
flatten/strided_slice/stackѕ
flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
flatten/strided_slice/stack_1ѕ
flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
flatten/strided_slice/stack_2њ
flatten/strided_sliceStridedSliceflatten/Shape:output:0$flatten/strided_slice/stack:output:0&flatten/strided_slice/stack_1:output:0&flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
flatten/strided_slice}
flatten/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2
flatten/Reshape/shape/1д
flatten/Reshape/shapePackflatten/strided_slice:output:0 flatten/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
flatten/Reshape/shape»
flatten/ReshapeReshape%leaky_re_lu_4/LeakyRelu:activations:0flatten/Reshape/shape:output:0*
T0*0
_output_shapes
:                  2
flatten/Reshapeд
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ1*
dtype02
dense_1/MatMul/ReadVariableOpЮ
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulц
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpА
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/SigmoidВ
IdentityIdentitydense_1/Sigmoid:y:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:+                           ::::::::::::2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ЈB
П
H__inference_sequential_layer_call_and_return_conditional_losses_68630618
dense_input
dense_68630570 
batch_normalization_68630573 
batch_normalization_68630575 
batch_normalization_68630577 
batch_normalization_68630579
conv2d_transpose_68630584"
batch_normalization_1_68630587"
batch_normalization_1_68630589"
batch_normalization_1_68630591"
batch_normalization_1_68630593
conv2d_transpose_1_68630598"
batch_normalization_2_68630601"
batch_normalization_2_68630603"
batch_normalization_2_68630605"
batch_normalization_2_68630607
conv2d_transpose_2_68630612
conv2d_transpose_2_68630614
identityѕб+batch_normalization/StatefulPartitionedCallб-batch_normalization_1/StatefulPartitionedCallб-batch_normalization_2/StatefulPartitionedCallб(conv2d_transpose/StatefulPartitionedCallб*conv2d_transpose_1/StatefulPartitionedCallб*conv2d_transpose_2/StatefulPartitionedCallбdense/StatefulPartitionedCallѓ
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_68630570*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_686303172
dense/StatefulPartitionedCallх
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_68630573batch_normalization_68630575batch_normalization_68630577batch_normalization_68630579*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_686299642-
+batch_normalization/StatefulPartitionedCallЉ
leaky_re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђb* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_686303692
leaky_re_lu/PartitionedCall§
reshape/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_686303912
reshape/PartitionedCallП
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_68630584*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_686300062*
(conv2d_transpose/StatefulPartitionedCallУ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0batch_normalization_1_68630587batch_normalization_1_68630589batch_normalization_1_68630591batch_normalization_1_68630593*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_686301072/
-batch_normalization_1/StatefulPartitionedCall│
leaky_re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_686304422
leaky_re_lu_1/PartitionedCallЉ
dropout/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_686304672
dropout/PartitionedCallС
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_transpose_1_68630598*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_686301492,
*conv2d_transpose_1/StatefulPartitionedCallж
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0batch_normalization_2_68630601batch_normalization_2_68630603batch_normalization_2_68630605batch_normalization_2_68630607*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_686302502/
-batch_normalization_2/StatefulPartitionedCall▓
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_686305232
leaky_re_lu_2/PartitionedCallќ
dropout_1/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_686305482
dropout_1/PartitionedCallЁ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_transpose_2_68630612conv2d_transpose_2_68630614*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_686302962,
*conv2d_transpose_2/StatefulPartitionedCallн
IdentityIdentity3conv2d_transpose_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*k
_input_shapesZ
X:         ђ:::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:U Q
(
_output_shapes
:         ђ
%
_user_specified_namedense_input
Р	
Џ
/__inference_sequential_1_layer_call_fn_68633411

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_686318622
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:+                           ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╗0
╠
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_68633642

inputs
assignmovingavg_68633617
assignmovingavg_1_68633623)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesљ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ђb*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ђb2
moments/StopGradientЦ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ђb2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ђb*
	keep_dims(2
moments/varianceЂ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ђb*
squeeze_dims
 2
moments/SqueezeЅ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ђb*
squeeze_dims
 2
moments/Squeeze_1╬
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg/68633617*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg/decayќ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_68633617*
_output_shapes	
:ђb*
dtype02 
AssignMovingAvg/ReadVariableOpЗ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg/68633617*
_output_shapes	
:ђb2
AssignMovingAvg/subв
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg/68633617*
_output_shapes	
:ђb2
AssignMovingAvg/mul│
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_68633617AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg/68633617*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpн
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*-
_class#
!loc:@AssignMovingAvg_1/68633623*
_output_shapes
: *
dtype0*
valueB
 *
О#<2
AssignMovingAvg_1/decayю
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_68633623*
_output_shapes	
:ђb*
dtype02"
 AssignMovingAvg_1/ReadVariableOp■
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/68633623*
_output_shapes	
:ђb2
AssignMovingAvg_1/subш
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/68633623*
_output_shapes	
:ђb2
AssignMovingAvg_1/mul┐
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_68633623AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*-
_class#
!loc:@AssignMovingAvg_1/68633623*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2
batchnorm/add/yЃ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ђb2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/RsqrtЪ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/mul/ReadVariableOpє
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђb2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ђb2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/mul_2Њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:ђb*
dtype02
batchnorm/ReadVariableOpѓ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ђb2
batchnorm/subє
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђb2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         ђb2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђb::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         ђb
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┬
serving_default«
N
sequential_input:
"serving_default_sequential_input:0         ђ@
sequential_10
StatefulPartitionedCall:0         tensorflow/serving/predict:Б┬
юц
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
з_default_save_signature
+З&call_and_return_all_conditional_losses
ш__call__" А
_tf_keras_sequential▀А{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_input"}}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 256]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gaussian_noise_input"}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise", "trainable": false, "dtype": "float32", "stddev": 0.2}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": false, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_input"}}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 256]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gaussian_noise_input"}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise", "trainable": false, "dtype": "float32", "stddev": 0.2}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": false, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0.25}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Addons>Yogi", "config": {"name": "Yogi", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta1": 0.8999999761581421, "beta2": 0.9990000128746033, "epsilon": 0.0010000000474974513, "l1_regularization_strength": 0.0, "l2_regularization_strength": 0.0, "activation": "sign", "initial_accumulator_value": 1e-06}}}}
Ѓ]
	layer_with_weights-0
	layer-0

layer_with_weights-1

layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
layer_with_weights-4
layer-8
layer_with_weights-5
layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
	variables
trainable_variables
regularization_losses
	keras_api
+Ш&call_and_return_all_conditional_losses
э__call__"љY
_tf_keras_sequentialыX{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 256]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 256]}}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}}
СJ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
 layer_with_weights-3
 layer-6
!layer-7
"layer-8
#layer-9
$layer_with_weights-4
$layer-10
%	optimizer
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+Э&call_and_return_all_conditional_losses
щ__call__"▓G
_tf_keras_sequentialЊG{"class_name": "Sequential", "name": "sequential_1", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gaussian_noise_input"}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise", "trainable": false, "dtype": "float32", "stddev": 0.2}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": false, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gaussian_noise_input"}}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise", "trainable": false, "dtype": "float32", "stddev": 0.2}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": false, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0.25}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Addons>Yogi", "config": {"name": "Yogi", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta1": 0.8999999761581421, "beta2": 0.9990000128746033, "epsilon": 0.0010000000474974513, "l1_regularization_strength": 0.0, "l2_regularization_strength": 0.0, "activation": "sign", "initial_accumulator_value": 1e-06}}}}
Ч
*iter

+beta_1

,beta_2
	-decay
.epsilon
/l1_regularization_strength
0l2_regularization_strength
1learning_rate2v═3v╬4v¤7vл8vЛ9vм<vМ=vн>vНAvоBvО2mп3m┘4m┌7m█8m▄9mП<mя=m▀>mЯAmрBmР"
	optimizer
■
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19
F20
G21
H22
I23
J24
K25
L26
M27
N28"
trackable_list_wrapper
n
20
31
42
73
84
95
<6
=7
>8
A9
B10"
trackable_list_wrapper
 "
trackable_list_wrapper
╬
Olayer_regularization_losses
	variables
trainable_variables
Player_metrics
Qmetrics
Rnon_trainable_variables

Slayers
regularization_losses
ш__call__
з_default_save_signature
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
-
Щserving_default"
signature_map
р

2kernel
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+ч&call_and_return_all_conditional_losses
Ч__call__"─
_tf_keras_layerф{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
Х	
Xaxis
	3gamma
4beta
5moving_mean
6moving_variance
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
+§&call_and_return_all_conditional_losses
■__call__"Я
_tf_keras_layerк{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 12544}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12544]}}
▄
]	variables
^trainable_variables
_regularization_losses
`	keras_api
+ &call_and_return_all_conditional_losses
ђ__call__"╦
_tf_keras_layer▒{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
э
a	variables
btrainable_variables
cregularization_losses
d	keras_api
+Ђ&call_and_return_all_conditional_losses
ѓ__call__"Т
_tf_keras_layer╠{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 256]}}}
Ъ


7kernel
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
+Ѓ&call_and_return_all_conditional_losses
ё__call__"ѓ	
_tf_keras_layerУ{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 256]}}
╝	
iaxis
	8gamma
9beta
:moving_mean
;moving_variance
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
+Ё&call_and_return_all_conditional_losses
є__call__"Т
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 128]}}
Я
n	variables
otrainable_variables
pregularization_losses
q	keras_api
+Є&call_and_return_all_conditional_losses
ѕ__call__"¤
_tf_keras_layerх{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
с
r	variables
strainable_variables
tregularization_losses
u	keras_api
+Ѕ&call_and_return_all_conditional_losses
і__call__"м
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
б


<kernel
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
+І&call_and_return_all_conditional_losses
ї__call__"Ё	
_tf_keras_layerв{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 128]}}
╝	
zaxis
	=gamma
>beta
?moving_mean
@moving_variance
{	variables
|trainable_variables
}regularization_losses
~	keras_api
+Ї&call_and_return_all_conditional_losses
ј__call__"Т
_tf_keras_layer╠{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
с
	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
+Ј&call_and_return_all_conditional_losses
љ__call__"¤
_tf_keras_layerх{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
в
Ѓ	variables
ёtrainable_variables
Ёregularization_losses
є	keras_api
+Љ&call_and_return_all_conditional_losses
њ__call__"о
_tf_keras_layer╝{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
г


Akernel
Bbias
Є	variables
ѕtrainable_variables
Ѕregularization_losses
і	keras_api
+Њ&call_and_return_all_conditional_losses
ћ__call__"Ђ	
_tf_keras_layerу{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
ъ
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16"
trackable_list_wrapper
n
20
31
42
73
84
95
<6
=7
>8
A9
B10"
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Іlayer_regularization_losses
	variables
trainable_variables
їlayer_metrics
Їmetrics
јnon_trainable_variables
Јlayers
regularization_losses
э__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
▄
љ	variables
Љtrainable_variables
њregularization_losses
Њ	keras_api
+Ћ&call_and_return_all_conditional_losses
ќ__call__"К
_tf_keras_layerГ{"class_name": "GaussianNoise", "name": "gaussian_noise", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise", "trainable": false, "dtype": "float32", "stddev": 0.2}}
Ь


Ckernel
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
+Ќ&call_and_return_all_conditional_losses
ў__call__"═	
_tf_keras_layer│	{"class_name": "Conv2D", "name": "conv2d", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
├	
	ўaxis
	Dgamma
Ebeta
Fmoving_mean
Gmoving_variance
Ў	variables
џtrainable_variables
Џregularization_losses
ю	keras_api
+Ў&call_and_return_all_conditional_losses
џ__call__"У
_tf_keras_layer╬{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
Т
Ю	variables
ъtrainable_variables
Ъregularization_losses
а	keras_api
+Џ&call_and_return_all_conditional_losses
ю__call__"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}
ь
А	variables
бtrainable_variables
Бregularization_losses
ц	keras_api
+Ю&call_and_return_all_conditional_losses
ъ__call__"п
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_2", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
З	

Hkernel
Ц	variables
дtrainable_variables
Дregularization_losses
е	keras_api
+Ъ&call_and_return_all_conditional_losses
а__call__"М
_tf_keras_layer╣{"class_name": "Conv2D", "name": "conv2d_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
├	
	Еaxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
ф	variables
Фtrainable_variables
гregularization_losses
Г	keras_api
+А&call_and_return_all_conditional_losses
б__call__"У
_tf_keras_layer╬{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": false, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 128]}}
ь
«	variables
»trainable_variables
░regularization_losses
▒	keras_api
+Б&call_and_return_all_conditional_losses
ц__call__"п
_tf_keras_layerЙ{"class_name": "Dropout", "name": "dropout_3", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": false, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
Т
▓	variables
│trainable_variables
┤regularization_losses
х	keras_api
+Ц&call_and_return_all_conditional_losses
д__call__"Л
_tf_keras_layerи{"class_name": "LeakyReLU", "name": "leaky_re_lu_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4", "trainable": false, "dtype": "float32", "alpha": 0.30000001192092896}}
Ж
Х	variables
иtrainable_variables
Иregularization_losses
╣	keras_api
+Д&call_and_return_all_conditional_losses
е__call__"Н
_tf_keras_layer╗{"class_name": "Flatten", "name": "flatten", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
■

Mkernel
Nbias
║	variables
╗trainable_variables
╝regularization_losses
й	keras_api
+Е&call_and_return_all_conditional_losses
ф__call__"М
_tf_keras_layer╣{"class_name": "Dense", "name": "dense_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": false, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6272}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6272]}}
╚
	Йiter
┐beta_1
└beta_2

┴decay
┬epsilon
├l1_regularization_strength
─l2_regularization_strength
┼learning_rateCvсDvСEvтHvТIvуJvУMvжNvЖCmвDmВEmьHmЬIm№Jm­MmыNmЫ"
	optimizer
v
C0
D1
E2
F3
G4
H5
I6
J7
K8
L9
M10
N11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 кlayer_regularization_losses
&	variables
'trainable_variables
Кlayer_metrics
╚metrics
╔non_trainable_variables
╩layers
(regularization_losses
щ__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Yogi/iter
: (2Yogi/beta_1
: (2Yogi/beta_2
: (2
Yogi/decay
: (2Yogi/epsilon
):' (2Yogi/l1_regularization_strength
):' (2Yogi/l2_regularization_strength
: (2Yogi/learning_rate
 :
ђђb2dense/kernel
(:&ђb2batch_normalization/gamma
':%ђb2batch_normalization/beta
0:.ђb (2batch_normalization/moving_mean
4:2ђb (2#batch_normalization/moving_variance
3:1ђђ2conv2d_transpose/kernel
*:(ђ2batch_normalization_1/gamma
):'ђ2batch_normalization_1/beta
2:0ђ (2!batch_normalization_1/moving_mean
6:4ђ (2%batch_normalization_1/moving_variance
4:2@ђ2conv2d_transpose_1/kernel
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
3:1@2conv2d_transpose_2/kernel
%:#2conv2d_transpose_2/bias
':%@2conv2d/kernel
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
*:(@ђ2conv2d_1/kernel
*:(ђ2batch_normalization_4/gamma
):'ђ2batch_normalization_4/beta
2:0ђ (2!batch_normalization_4/moving_mean
6:4ђ (2%batch_normalization_4/moving_variance
!:	ђ12dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
╦0"
trackable_list_wrapper
д
50
61
:2
;3
?4
@5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
20"
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_list_wrapper
х
 ╠layer_regularization_losses
T	variables
Utrainable_variables
═layer_metrics
╬metrics
¤non_trainable_variables
лlayers
Vregularization_losses
Ч__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Лlayer_regularization_losses
Y	variables
Ztrainable_variables
мlayer_metrics
Мmetrics
нnon_trainable_variables
Нlayers
[regularization_losses
■__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 оlayer_regularization_losses
]	variables
^trainable_variables
Оlayer_metrics
пmetrics
┘non_trainable_variables
┌layers
_regularization_losses
ђ__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 █layer_regularization_losses
a	variables
btrainable_variables
▄layer_metrics
Пmetrics
яnon_trainable_variables
▀layers
cregularization_losses
ѓ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Яlayer_regularization_losses
e	variables
ftrainable_variables
рlayer_metrics
Рmetrics
сnon_trainable_variables
Сlayers
gregularization_losses
ё__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
х
 тlayer_regularization_losses
j	variables
ktrainable_variables
Тlayer_metrics
уmetrics
Уnon_trainable_variables
жlayers
lregularization_losses
є__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Жlayer_regularization_losses
n	variables
otrainable_variables
вlayer_metrics
Вmetrics
ьnon_trainable_variables
Ьlayers
pregularization_losses
ѕ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
 №layer_regularization_losses
r	variables
strainable_variables
­layer_metrics
ыmetrics
Ыnon_trainable_variables
зlayers
tregularization_losses
і__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
х
 Зlayer_regularization_losses
v	variables
wtrainable_variables
шlayer_metrics
Шmetrics
эnon_trainable_variables
Эlayers
xregularization_losses
ї__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
=0
>1
?2
@3"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
 щlayer_regularization_losses
{	variables
|trainable_variables
Щlayer_metrics
чmetrics
Чnon_trainable_variables
§layers
}regularization_losses
ј__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
и
 ■layer_regularization_losses
	variables
ђtrainable_variables
 layer_metrics
ђmetrics
Ђnon_trainable_variables
ѓlayers
Ђregularization_losses
љ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Ѓlayer_regularization_losses
Ѓ	variables
ёtrainable_variables
ёlayer_metrics
Ёmetrics
єnon_trainable_variables
Єlayers
Ёregularization_losses
њ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
 ѕlayer_regularization_losses
Є	variables
ѕtrainable_variables
Ѕlayer_metrics
іmetrics
Іnon_trainable_variables
їlayers
Ѕregularization_losses
ћ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
50
61
:2
;3
?4
@5"
trackable_list_wrapper
~
	0

1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Їlayer_regularization_losses
љ	variables
Љtrainable_variables
јlayer_metrics
Јmetrics
љnon_trainable_variables
Љlayers
њregularization_losses
ќ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 њlayer_regularization_losses
ћ	variables
Ћtrainable_variables
Њlayer_metrics
ћmetrics
Ћnon_trainable_variables
ќlayers
ќregularization_losses
ў__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
D0
E1
F2
G3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Ќlayer_regularization_losses
Ў	variables
џtrainable_variables
ўlayer_metrics
Ўmetrics
џnon_trainable_variables
Џlayers
Џregularization_losses
џ__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 юlayer_regularization_losses
Ю	variables
ъtrainable_variables
Юlayer_metrics
ъmetrics
Ъnon_trainable_variables
аlayers
Ъregularization_losses
ю__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Аlayer_regularization_losses
А	variables
бtrainable_variables
бlayer_metrics
Бmetrics
цnon_trainable_variables
Цlayers
Бregularization_losses
ъ__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 дlayer_regularization_losses
Ц	variables
дtrainable_variables
Дlayer_metrics
еmetrics
Еnon_trainable_variables
фlayers
Дregularization_losses
а__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
I0
J1
K2
L3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 Фlayer_regularization_losses
ф	variables
Фtrainable_variables
гlayer_metrics
Гmetrics
«non_trainable_variables
»layers
гregularization_losses
б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 ░layer_regularization_losses
«	variables
»trainable_variables
▒layer_metrics
▓metrics
│non_trainable_variables
┤layers
░regularization_losses
ц__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 хlayer_regularization_losses
▓	variables
│trainable_variables
Хlayer_metrics
иmetrics
Иnon_trainable_variables
╣layers
┤regularization_losses
д__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 ║layer_regularization_losses
Х	variables
иtrainable_variables
╗layer_metrics
╝metrics
йnon_trainable_variables
Йlayers
Иregularization_losses
е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 ┐layer_regularization_losses
║	variables
╗trainable_variables
└layer_metrics
┴metrics
┬non_trainable_variables
├layers
╝regularization_losses
ф__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Yogi/iter
: (2Yogi/beta_1
: (2Yogi/beta_2
: (2
Yogi/decay
: (2Yogi/epsilon
):' (2Yogi/l1_regularization_strength
):' (2Yogi/l2_regularization_strength
: (2Yogi/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
─0"
trackable_list_wrapper
v
C0
D1
E2
F3
G4
H5
I6
J7
K8
L9
M10
N11"
trackable_list_wrapper
n
0
1
2
3
4
5
 6
!7
"8
#9
$10"
trackable_list_wrapper
┐

┼total

кcount
К	variables
╚	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
.
50
61"
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
.
:0
;1"
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
.
?0
@1"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
<
D0
E1
F2
G3"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
I0
J1
K2
L3"
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
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
┐

╔total

╩count
╦	variables
╠	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
┼0
к1"
trackable_list_wrapper
.
К	variables"
_generic_user_object
:  (2total
:  (2count
0
╔0
╩1"
trackable_list_wrapper
.
╦	variables"
_generic_user_object
%:#
ђђb2Yogi/dense/kernel/v
-:+ђb2 Yogi/batch_normalization/gamma/v
,:*ђb2Yogi/batch_normalization/beta/v
8:6ђђ2Yogi/conv2d_transpose/kernel/v
/:-ђ2"Yogi/batch_normalization_1/gamma/v
.:,ђ2!Yogi/batch_normalization_1/beta/v
9:7@ђ2 Yogi/conv2d_transpose_1/kernel/v
.:,@2"Yogi/batch_normalization_2/gamma/v
-:+@2!Yogi/batch_normalization_2/beta/v
8:6@2 Yogi/conv2d_transpose_2/kernel/v
*:(2Yogi/conv2d_transpose_2/bias/v
%:#
ђђb2Yogi/dense/kernel/m
-:+ђb2 Yogi/batch_normalization/gamma/m
,:*ђb2Yogi/batch_normalization/beta/m
8:6ђђ2Yogi/conv2d_transpose/kernel/m
/:-ђ2"Yogi/batch_normalization_1/gamma/m
.:,ђ2!Yogi/batch_normalization_1/beta/m
9:7@ђ2 Yogi/conv2d_transpose_1/kernel/m
.:,@2"Yogi/batch_normalization_2/gamma/m
-:+@2!Yogi/batch_normalization_2/beta/m
8:6@2 Yogi/conv2d_transpose_2/kernel/m
*:(2Yogi/conv2d_transpose_2/bias/m
,:*@2Yogi/conv2d/kernel/v
.:,@2"Yogi/batch_normalization_3/gamma/v
-:+@2!Yogi/batch_normalization_3/beta/v
/:-@ђ2Yogi/conv2d_1/kernel/v
/:-ђ2"Yogi/batch_normalization_4/gamma/v
.:,ђ2!Yogi/batch_normalization_4/beta/v
&:$	ђ12Yogi/dense_1/kernel/v
:2Yogi/dense_1/bias/v
,:*@2Yogi/conv2d/kernel/m
.:,@2"Yogi/batch_normalization_3/gamma/m
-:+@2!Yogi/batch_normalization_3/beta/m
/:-@ђ2Yogi/conv2d_1/kernel/m
/:-ђ2"Yogi/batch_normalization_4/gamma/m
.:,ђ2!Yogi/batch_normalization_4/beta/m
&:$	ђ12Yogi/dense_1/kernel/m
:2Yogi/dense_1/bias/m
в2У
#__inference__wrapped_model_68629835└
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *0б-
+і(
sequential_input         ђ
Ш2з
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632565
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632011
J__inference_sequential_2_layer_call_and_return_conditional_losses_68631947
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632730└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
і2Є
/__inference_sequential_2_layer_call_fn_68632139
/__inference_sequential_2_layer_call_fn_68632856
/__inference_sequential_2_layer_call_fn_68632793
/__inference_sequential_2_layer_call_fn_68632266└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
H__inference_sequential_layer_call_and_return_conditional_losses_68630618
H__inference_sequential_layer_call_and_return_conditional_losses_68633008
H__inference_sequential_layer_call_and_return_conditional_losses_68633126
H__inference_sequential_layer_call_and_return_conditional_losses_68630567└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ѓ2 
-__inference_sequential_layer_call_fn_68630709
-__inference_sequential_layer_call_fn_68633165
-__inference_sequential_layer_call_fn_68633204
-__inference_sequential_layer_call_fn_68630799└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ј2І
J__inference_sequential_1_layer_call_and_return_conditional_losses_68633353
J__inference_sequential_1_layer_call_and_return_conditional_losses_68633483
J__inference_sequential_1_layer_call_and_return_conditional_losses_68631339
J__inference_sequential_1_layer_call_and_return_conditional_losses_68631378
J__inference_sequential_1_layer_call_and_return_conditional_losses_68633296
J__inference_sequential_1_layer_call_and_return_conditional_losses_68633534└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
В2ж
/__inference_sequential_1_layer_call_fn_68633411
/__inference_sequential_1_layer_call_fn_68633563
/__inference_sequential_1_layer_call_fn_68633382
/__inference_sequential_1_layer_call_fn_68631515
/__inference_sequential_1_layer_call_fn_68631447
/__inference_sequential_1_layer_call_fn_68633592└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
оBМ
&__inference_signature_wrapper_68632345sequential_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_layer_call_and_return_conditional_losses_68633599б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_layer_call_fn_68633606б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Я2П
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_68633642
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_68633662┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ф2Д
6__inference_batch_normalization_layer_call_fn_68633688
6__inference_batch_normalization_layer_call_fn_68633675┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
з2­
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_68633693б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_leaky_re_lu_layer_call_fn_68633698б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_reshape_layer_call_and_return_conditional_losses_68633712б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_reshape_layer_call_fn_68633717б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
«2Ф
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_68630006п
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *8б5
3і0,                           ђ
Њ2љ
3__inference_conv2d_transpose_layer_call_fn_68630014п
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *8б5
3і0,                           ђ
С2р
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68633755
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68633737┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
«2Ф
8__inference_batch_normalization_1_layer_call_fn_68633781
8__inference_batch_normalization_1_layer_call_fn_68633768┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_68633786б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
0__inference_leaky_re_lu_1_layer_call_fn_68633791б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╚2┼
E__inference_dropout_layer_call_and_return_conditional_losses_68633803
E__inference_dropout_layer_call_and_return_conditional_losses_68633808┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
*__inference_dropout_layer_call_fn_68633818
*__inference_dropout_layer_call_fn_68633813┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
░2Г
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_68630149п
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *8б5
3і0,                           ђ
Ћ2њ
5__inference_conv2d_transpose_1_layer_call_fn_68630157п
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *8б5
3і0,                           ђ
С2р
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68633838
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68633856┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
«2Ф
8__inference_batch_normalization_2_layer_call_fn_68633882
8__inference_batch_normalization_2_layer_call_fn_68633869┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_68633887б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
0__inference_leaky_re_lu_2_layer_call_fn_68633892б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠2╔
G__inference_dropout_1_layer_call_and_return_conditional_losses_68633909
G__inference_dropout_1_layer_call_and_return_conditional_losses_68633904┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ќ2Њ
,__inference_dropout_1_layer_call_fn_68633914
,__inference_dropout_1_layer_call_fn_68633919┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
»2г
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_68630296О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
ћ2Љ
5__inference_conv2d_transpose_2_layer_call_fn_68630306О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
о2М
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_68633930
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_68633934┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
а2Ю
1__inference_gaussian_noise_layer_call_fn_68633939
1__inference_gaussian_noise_layer_call_fn_68633944┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
D__inference_conv2d_layer_call_and_return_conditional_losses_68633951б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_conv2d_layer_call_fn_68633958б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ј2І
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68633976
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68634056
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68633994
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68634038┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
б2Ъ
8__inference_batch_normalization_3_layer_call_fn_68634082
8__inference_batch_normalization_3_layer_call_fn_68634020
8__inference_batch_normalization_3_layer_call_fn_68634007
8__inference_batch_normalization_3_layer_call_fn_68634069┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_68634087б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
0__inference_leaky_re_lu_3_layer_call_fn_68634092б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠2╔
G__inference_dropout_2_layer_call_and_return_conditional_losses_68634109
G__inference_dropout_2_layer_call_and_return_conditional_losses_68634104┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ќ2Њ
,__inference_dropout_2_layer_call_fn_68634114
,__inference_dropout_2_layer_call_fn_68634119┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_conv2d_1_layer_call_and_return_conditional_losses_68634126б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_conv2d_1_layer_call_fn_68634133б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ј2І
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634213
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634169
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634151
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634231┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
б2Ъ
8__inference_batch_normalization_4_layer_call_fn_68634195
8__inference_batch_normalization_4_layer_call_fn_68634244
8__inference_batch_normalization_4_layer_call_fn_68634182
8__inference_batch_normalization_4_layer_call_fn_68634257┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╠2╔
G__inference_dropout_3_layer_call_and_return_conditional_losses_68634274
G__inference_dropout_3_layer_call_and_return_conditional_losses_68634269┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ќ2Њ
,__inference_dropout_3_layer_call_fn_68634279
,__inference_dropout_3_layer_call_fn_68634284┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ш2Ы
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_68634289б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
0__inference_leaky_re_lu_4_layer_call_fn_68634294б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_flatten_layer_call_and_return_conditional_losses_68634300б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_flatten_layer_call_fn_68634305б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_dense_1_layer_call_and_return_conditional_losses_68634316б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_dense_1_layer_call_fn_68634325б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 └
#__inference__wrapped_model_68629835ў26354789:;<=>?@ABCDEFGHIJKLMN:б7
0б-
+і(
sequential_input         ђ
ф ";ф8
6
sequential_1&і#
sequential_1         ­
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68633737ў89:;NбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ­
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68633755ў89:;NбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ ╚
8__inference_batch_normalization_1_layer_call_fn_68633768І89:;NбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђ╚
8__inference_batch_normalization_1_layer_call_fn_68633781І89:;NбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђЬ
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68633838ќ=>?@MбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ Ь
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68633856ќ=>?@MбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ к
8__inference_batch_normalization_2_layer_call_fn_68633869Ѕ=>?@MбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @к
8__inference_batch_normalization_2_layer_call_fn_68633882Ѕ=>?@MбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @Ь
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68633976ќDEFGMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ Ь
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68633994ќDEFGMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ ╔
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68634038rDEFG;б8
1б.
(і%
inputs         @
p
ф "-б*
#і 
0         @
џ ╔
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68634056rDEFG;б8
1б.
(і%
inputs         @
p 
ф "-б*
#і 
0         @
џ к
8__inference_batch_normalization_3_layer_call_fn_68634007ЅDEFGMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @к
8__inference_batch_normalization_3_layer_call_fn_68634020ЅDEFGMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @А
8__inference_batch_normalization_3_layer_call_fn_68634069eDEFG;б8
1б.
(і%
inputs         @
p
ф " і         @А
8__inference_batch_normalization_3_layer_call_fn_68634082eDEFG;б8
1б.
(і%
inputs         @
p 
ф " і         @­
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634151ўIJKLNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ­
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634169ўIJKLNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ ╦
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634213tIJKL<б9
2б/
)і&
inputs         ђ
p
ф ".б+
$і!
0         ђ
џ ╦
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_68634231tIJKL<б9
2б/
)і&
inputs         ђ
p 
ф ".б+
$і!
0         ђ
џ ╚
8__inference_batch_normalization_4_layer_call_fn_68634182ІIJKLNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђ╚
8__inference_batch_normalization_4_layer_call_fn_68634195ІIJKLNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђБ
8__inference_batch_normalization_4_layer_call_fn_68634244gIJKL<б9
2б/
)і&
inputs         ђ
p
ф "!і         ђБ
8__inference_batch_normalization_4_layer_call_fn_68634257gIJKL<б9
2б/
)і&
inputs         ђ
p 
ф "!і         ђ╣
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_68633642d56344б1
*б'
!і
inputs         ђb
p
ф "&б#
і
0         ђb
џ ╣
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_68633662d63544б1
*б'
!і
inputs         ђb
p 
ф "&б#
і
0         ђb
џ Љ
6__inference_batch_normalization_layer_call_fn_68633675W56344б1
*б'
!і
inputs         ђb
p
ф "і         ђbЉ
6__inference_batch_normalization_layer_call_fn_68633688W63544б1
*б'
!і
inputs         ђb
p 
ф "і         ђbХ
F__inference_conv2d_1_layer_call_and_return_conditional_losses_68634126lH7б4
-б*
(і%
inputs         @
ф ".б+
$і!
0         ђ
џ ј
+__inference_conv2d_1_layer_call_fn_68634133_H7б4
-б*
(і%
inputs         @
ф "!і         ђ│
D__inference_conv2d_layer_call_and_return_conditional_losses_68633951kC7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         @
џ І
)__inference_conv2d_layer_call_fn_68633958^C7б4
-б*
(і%
inputs         
ф " і         @т
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_68630149љ<JбG
@б=
;і8
inputs,                           ђ
ф "?б<
5і2
0+                           @
џ й
5__inference_conv2d_transpose_1_layer_call_fn_68630157Ѓ<JбG
@б=
;і8
inputs,                           ђ
ф "2і/+                           @т
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_68630296љABIбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                           
џ й
5__inference_conv2d_transpose_2_layer_call_fn_68630306ЃABIбF
?б<
:і7
inputs+                           @
ф "2і/+                           С
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_68630006Љ7JбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ ╝
3__inference_conv2d_transpose_layer_call_fn_68630014ё7JбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђд
E__inference_dense_1_layer_call_and_return_conditional_losses_68634316]MN0б-
&б#
!і
inputs         ђ1
ф "%б"
і
0         
џ ~
*__inference_dense_1_layer_call_fn_68634325PMN0б-
&б#
!і
inputs         ђ1
ф "і         ц
C__inference_dense_layer_call_and_return_conditional_losses_68633599]20б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђb
џ |
(__inference_dense_layer_call_fn_68633606P20б-
&б#
!і
inputs         ђ
ф "і         ђb▄
G__inference_dropout_1_layer_call_and_return_conditional_losses_68633904љMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ▄
G__inference_dropout_1_layer_call_and_return_conditional_losses_68633909љMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ ┤
,__inference_dropout_1_layer_call_fn_68633914ЃMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @┤
,__inference_dropout_1_layer_call_fn_68633919ЃMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @и
G__inference_dropout_2_layer_call_and_return_conditional_losses_68634104l;б8
1б.
(і%
inputs         @
p
ф "-б*
#і 
0         @
џ и
G__inference_dropout_2_layer_call_and_return_conditional_losses_68634109l;б8
1б.
(і%
inputs         @
p 
ф "-б*
#і 
0         @
џ Ј
,__inference_dropout_2_layer_call_fn_68634114_;б8
1б.
(і%
inputs         @
p
ф " і         @Ј
,__inference_dropout_2_layer_call_fn_68634119_;б8
1б.
(і%
inputs         @
p 
ф " і         @╣
G__inference_dropout_3_layer_call_and_return_conditional_losses_68634269n<б9
2б/
)і&
inputs         ђ
p
ф ".б+
$і!
0         ђ
џ ╣
G__inference_dropout_3_layer_call_and_return_conditional_losses_68634274n<б9
2б/
)і&
inputs         ђ
p 
ф ".б+
$і!
0         ђ
џ Љ
,__inference_dropout_3_layer_call_fn_68634279a<б9
2б/
)і&
inputs         ђ
p
ф "!і         ђЉ
,__inference_dropout_3_layer_call_fn_68634284a<б9
2б/
)і&
inputs         ђ
p 
ф "!і         ђ▄
E__inference_dropout_layer_call_and_return_conditional_losses_68633803њNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ▄
E__inference_dropout_layer_call_and_return_conditional_losses_68633808њNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ ┤
*__inference_dropout_layer_call_fn_68633813ЁNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђ┤
*__inference_dropout_layer_call_fn_68633818ЁNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђФ
E__inference_flatten_layer_call_and_return_conditional_losses_68634300b8б5
.б+
)і&
inputs         ђ
ф "&б#
і
0         ђ1
џ Ѓ
*__inference_flatten_layer_call_fn_68634305U8б5
.б+
)і&
inputs         ђ
ф "і         ђ1╝
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_68633930l;б8
1б.
(і%
inputs         
p
ф "-б*
#і 
0         
џ ╝
L__inference_gaussian_noise_layer_call_and_return_conditional_losses_68633934l;б8
1б.
(і%
inputs         
p 
ф "-б*
#і 
0         
џ ћ
1__inference_gaussian_noise_layer_call_fn_68633939_;б8
1б.
(і%
inputs         
p
ф " і         ћ
1__inference_gaussian_noise_layer_call_fn_68633944_;б8
1б.
(і%
inputs         
p 
ф " і         я
K__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_68633786јJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ Х
0__inference_leaky_re_lu_1_layer_call_fn_68633791ЂJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђ▄
K__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_68633887їIбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                           @
џ │
0__inference_leaky_re_lu_2_layer_call_fn_68633892IбF
?б<
:і7
inputs+                           @
ф "2і/+                           @и
K__inference_leaky_re_lu_3_layer_call_and_return_conditional_losses_68634087h7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ Ј
0__inference_leaky_re_lu_3_layer_call_fn_68634092[7б4
-б*
(і%
inputs         @
ф " і         @╣
K__inference_leaky_re_lu_4_layer_call_and_return_conditional_losses_68634289j8б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ Љ
0__inference_leaky_re_lu_4_layer_call_fn_68634294]8б5
.б+
)і&
inputs         ђ
ф "!і         ђД
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_68633693Z0б-
&б#
!і
inputs         ђb
ф "&б#
і
0         ђb
џ 
.__inference_leaky_re_lu_layer_call_fn_68633698M0б-
&б#
!і
inputs         ђb
ф "і         ђbФ
E__inference_reshape_layer_call_and_return_conditional_losses_68633712b0б-
&б#
!і
inputs         ђb
ф ".б+
$і!
0         ђ
џ Ѓ
*__inference_reshape_layer_call_fn_68633717U0б-
&б#
!і
inputs         ђb
ф "!і         ђМ
J__inference_sequential_1_layer_call_and_return_conditional_losses_68631339ёCDEFGHIJKLMNMбJ
Cб@
6і3
gaussian_noise_input         
p

 
ф "%б"
і
0         
џ М
J__inference_sequential_1_layer_call_and_return_conditional_losses_68631378ёCDEFGHIJKLMNMбJ
Cб@
6і3
gaussian_noise_input         
p 

 
ф "%б"
і
0         
џ О
J__inference_sequential_1_layer_call_and_return_conditional_losses_68633296ѕCDEFGHIJKLMNQбN
GбD
:і7
inputs+                           
p

 
ф "%б"
і
0         
џ О
J__inference_sequential_1_layer_call_and_return_conditional_losses_68633353ѕCDEFGHIJKLMNQбN
GбD
:і7
inputs+                           
p 

 
ф "%б"
і
0         
џ ─
J__inference_sequential_1_layer_call_and_return_conditional_losses_68633483vCDEFGHIJKLMN?б<
5б2
(і%
inputs         
p

 
ф "%б"
і
0         
џ ─
J__inference_sequential_1_layer_call_and_return_conditional_losses_68633534vCDEFGHIJKLMN?б<
5б2
(і%
inputs         
p 

 
ф "%б"
і
0         
џ ф
/__inference_sequential_1_layer_call_fn_68631447wCDEFGHIJKLMNMбJ
Cб@
6і3
gaussian_noise_input         
p

 
ф "і         ф
/__inference_sequential_1_layer_call_fn_68631515wCDEFGHIJKLMNMбJ
Cб@
6і3
gaussian_noise_input         
p 

 
ф "і         «
/__inference_sequential_1_layer_call_fn_68633382{CDEFGHIJKLMNQбN
GбD
:і7
inputs+                           
p

 
ф "і         «
/__inference_sequential_1_layer_call_fn_68633411{CDEFGHIJKLMNQбN
GбD
:і7
inputs+                           
p 

 
ф "і         ю
/__inference_sequential_1_layer_call_fn_68633563iCDEFGHIJKLMN?б<
5б2
(і%
inputs         
p

 
ф "і         ю
/__inference_sequential_1_layer_call_fn_68633592iCDEFGHIJKLMN?б<
5б2
(і%
inputs         
p 

 
ф "і         ┘
J__inference_sequential_2_layer_call_and_return_conditional_losses_68631947і25634789:;<=>?@ABCDEFGHIJKLMNBб?
8б5
+і(
sequential_input         ђ
p

 
ф "%б"
і
0         
џ ┘
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632011і26354789:;<=>?@ABCDEFGHIJKLMNBб?
8б5
+і(
sequential_input         ђ
p 

 
ф "%б"
і
0         
џ ¤
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632565ђ25634789:;<=>?@ABCDEFGHIJKLMN8б5
.б+
!і
inputs         ђ
p

 
ф "%б"
і
0         
џ ¤
J__inference_sequential_2_layer_call_and_return_conditional_losses_68632730ђ26354789:;<=>?@ABCDEFGHIJKLMN8б5
.б+
!і
inputs         ђ
p 

 
ф "%б"
і
0         
џ ░
/__inference_sequential_2_layer_call_fn_68632139}25634789:;<=>?@ABCDEFGHIJKLMNBб?
8б5
+і(
sequential_input         ђ
p

 
ф "і         ░
/__inference_sequential_2_layer_call_fn_68632266}26354789:;<=>?@ABCDEFGHIJKLMNBб?
8б5
+і(
sequential_input         ђ
p 

 
ф "і         д
/__inference_sequential_2_layer_call_fn_68632793s25634789:;<=>?@ABCDEFGHIJKLMN8б5
.б+
!і
inputs         ђ
p

 
ф "і         д
/__inference_sequential_2_layer_call_fn_68632856s26354789:;<=>?@ABCDEFGHIJKLMN8б5
.б+
!і
inputs         ђ
p 

 
ф "і         Я
H__inference_sequential_layer_call_and_return_conditional_losses_68630567Њ25634789:;<=>?@AB=б:
3б0
&і#
dense_input         ђ
p

 
ф "?б<
5і2
0+                           
џ Я
H__inference_sequential_layer_call_and_return_conditional_losses_68630618Њ26354789:;<=>?@AB=б:
3б0
&і#
dense_input         ђ
p 

 
ф "?б<
5і2
0+                           
џ ╚
H__inference_sequential_layer_call_and_return_conditional_losses_68633008|25634789:;<=>?@AB8б5
.б+
!і
inputs         ђ
p

 
ф "-б*
#і 
0         
џ ╚
H__inference_sequential_layer_call_and_return_conditional_losses_68633126|26354789:;<=>?@AB8б5
.б+
!і
inputs         ђ
p 

 
ф "-б*
#і 
0         
џ И
-__inference_sequential_layer_call_fn_68630709є25634789:;<=>?@AB=б:
3б0
&і#
dense_input         ђ
p

 
ф "2і/+                           И
-__inference_sequential_layer_call_fn_68630799є26354789:;<=>?@AB=б:
3б0
&і#
dense_input         ђ
p 

 
ф "2і/+                           │
-__inference_sequential_layer_call_fn_68633165Ђ25634789:;<=>?@AB8б5
.б+
!і
inputs         ђ
p

 
ф "2і/+                           │
-__inference_sequential_layer_call_fn_68633204Ђ26354789:;<=>?@AB8б5
.б+
!і
inputs         ђ
p 

 
ф "2і/+                           О
&__inference_signature_wrapper_68632345г26354789:;<=>?@ABCDEFGHIJKLMNNбK
б 
DфA
?
sequential_input+і(
sequential_input         ђ";ф8
6
sequential_1&і#
sequential_1         