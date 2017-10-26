## List of TensorFlow Ops that have valid gradients

AccumulatorApplyGradient =>	Applies a gradient to a given accumulator.

AccumulatorNumAccumulated =>	Returns the number of gradients aggregated in the given accumulators.

AccumulatorSetGlobalStep =>	Updates the accumulator with a new value for global_step.

AccumulatorTakeGradient =>	Extracts the average gradient in the given ConditionalAccumulator.

Add =>	Returns x + y element-wise.

AudioSummary =>	Outputs a `Summary` protocol buffer with audio.

AudioSummaryV2 =>	Outputs a `Summary` protocol buffer with audio.

BarrierClose =>	Closes the given barrier.

BarrierIncompleteSize =>	Computes the number of incomplete elements in the given barrier.

BarrierInsertMany =>	For each key, assigns the respective value to the specified component.

BarrierReadySize =>	Computes the number of complete elements in the given barrier.

BarrierTakeMany =>	Takes the given number of completed elements from a barrier.

DecodeBase64 =>	Decode web-safe base64-encoded strings.

DecodeBmp =>	Decode the first frame of a BMP-encoded image to a uint8 tensor.

DecodeCSV =>	Convert CSV records to tensors. Each column maps to one tensor.

DecodeGif =>	Decode the first frame of a GIF-encoded image to a uint8 tensor.

DecodeJSONExample =>	Convert JSON-encoded Example records to binary protocol buffer strings.

DecodeJpeg =>	Decode a JPEG-encoded image to a uint8 tensor.

DecodePng =>	Decode a PNG-encoded image to a uint8 or uint16 tensor.

DecodeRaw =>	Reinterpret the bytes of a string as a vector of numbers.

DecodeWav =>	Decode a 16-bit PCM WAV file to a float tensor.

DeleteSessionTensor =>	Delete the tensor specified by its handle in the session.

DenseToDenseSetOperation =>	Applies set operation along last dimension of 2 `Tensor` inputs.

DenseToSparseSetOperation =>	Applies set operation along last dimension of `Tensor` and `SparseTensor`.

DeserializeManySparse =>	Deserialize and concatenate `SparseTensors` from a serialized minibatch.

EncodeBase64 =>	Encode strings into web-safe base64 format.

Equal =>	Returns the truth value of (x == y) element-wise.

FixedLengthRecordDataset =>	Creates a dataset that emits the records from one or more binary files.

GetSessionTensor =>	Get the value of the tensor specified by its handle.

HistogramSummary =>	Outputs a `Summary` protocol buffer with a histogram.

ImageSummary =>	Outputs a `Summary` protocol buffer with images.

InitializeTable =>	Table initializer that takes two tensors for keys and values respectively.

InitializeTableFromTextFile =>	Initializes a table from a text file.

InitializeTableFromTextFileV2 =>	Initializes a table from a text file.

LookupTableExport =>	Outputs all keys and values in the table.

LookupTableFind =>	Looks up keys in a table, outputs the corresponding values.

LookupTableImport =>	Replaces the contents of the table with the specified keys and values.

LookupTableInsert =>	Updates the table to associates keys with values.

LookupTableSize =>	Computes the number of elements in the given table.

MatchingFiles =>	Returns the set of files matching one or more glob patterns.

MergeSummary =>	Merges summaries.

MergeV2Checkpoints =>	V2 format specific: merges the metadata files of sharded checkpoints.  The

NotEqual =>	Returns the truth value of (x != y) element-wise.

ParseExample =>	Transforms a vector of brain.Example protos (as strings) into typed tensors.

ParseSingleSequenceExample =>	Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.

ParseTensor =>	Transforms a serialized tensorflow.TensorProto proto into a Tensor.

QueueClose =>	Closes the given queue.

QueueDequeue =>	Dequeues a tuple of one or more tensors from the given queue.

QueueDequeueMany =>	Dequeues `n` tuples of one or more tensors from the given queue.

QueueDequeueUpTo =>	Dequeues `n` tuples of one or more tensors from the given queue.

QueueEnqueue =>	Enqueues a tuple of one or more tensors in the given queue.

QueueEnqueueMany =>	Enqueues zero or more tuples of one or more tensors in the given queue.

QueueSize =>	Computes the number of elements in the given queue.

ReadFile =>	Reads and outputs the entire contents of the input filename.

ReaderNumRecordsProduced =>	Returns the number of records this Reader has produced.

ReaderNumWorkUnitsCompleted =>	Returns the number of work units this Reader has finished processing.

ReaderRead =>	Returns the next record (key, value pair) produced by a Reader.

ReaderReadUpTo =>	Returns up to `num_records` (key, value) pairs produced by a Reader.

ReaderReset =>	Restore a Reader to its initial clean state.

ReaderRestoreState =>	Restore a reader to a previously saved state.

ReaderRestoreStateV2 =>	Restore a reader to a previously saved state.

ReaderSerializeState =>	Produce a string tensor that encodes the state of a Reader.

ReduceJoin =>	Joins a string Tensor across the given dimensions.

Restore =>	Restores a tensor from checkpoint files.

RestoreSlice =>	Restores a tensor from checkpoint files.

RestoreV2 =>	Restores tensors from a V2 checkpoint.

Reverse =>	Reverses specific dimensions of a tensor.

Save =>	Saves the input tensors to disk.

SaveSlices =>	Saves input tensors slices to disk.

SaveV2 =>	Saves tensors in V2 checkpoint format.

ScalarSummary =>	Outputs a `Summary` protocol buffer with scalar values.

SdcaFprint =>	Computes fingerprints of the input strings.

SetSize =>	Number of unique elements along last dimension of input `set`.

ShardedFilename =>	Generate a sharded filename. The filename is printf formatted as

ShardedFilespec =>	Generate a glob pattern matching all sharded file names.

SparseAccumulatorApplyGradient =>	Applies a sparse gradient to a given accumulator.

SparseAccumulatorTakeGradient =>	Extracts the average sparse gradient in a SparseConditionalAccumulator.

SparseCross =>	Generates sparse cross from a list of sparse and dense tensors.

SparseToSparseSetOperation =>	Applies set operation along last dimension of 2 `SparseTensor` inputs.

StackClose =>	Delete the stack from its resource container.

StackPop =>	Pop the element at the top of the stack.

StackPush =>	Push an element onto the stack.

StringJoin =>	Joins the strings in the given list of string tensors into one tensor;

StringSplit =>	Split elements of `input` based on `delimiter` into a `SparseTensor`.

StringToHashBucket =>	Converts each string in the input Tensor to its hash mod by a number of buckets.

StringToHashBucketFast =>	Converts each string in the input Tensor to its hash mod by a number of buckets.

StringToHashBucketStrong =>	Converts each string in the input Tensor to its hash mod by a number of buckets.

StringToNumber =>	Converts each string in the input Tensor to the specified numeric type.

Substr =>	Return substrings from `Tensor` of strings.

TFRecordDataset =>	Creates a dataset that emits the records from one or more TFRecord files.

TensorArrayClose =>	

TensorArrayCloseV2 =>	Deprecated. Use TensorArrayCloseV3

TensorArrayConcat =>	

TensorArrayConcatV2 =>	Deprecated. Use TensorArrayConcatV3

TensorArrayGather =>	

TensorArrayGatherV2 =>	Deprecated. Use TensorArrayGatherV3

TensorArrayGrad =>	

TensorArrayGradV2 =>	Deprecated. Use TensorArrayGradV3

TensorArrayPack =>	

TensorArrayRead =>	

TensorArrayReadV2 =>	Deprecated. Use TensorArrayReadV3

TensorArrayScatter =>	

TensorArrayScatterV2 =>	Deprecated. Use TensorArrayScatterV3

TensorArraySize =>	

TensorArraySizeV2 =>	Deprecated. Use TensorArraySizeV3

TensorArraySplit =>	

TensorArraySplitV2 =>	Deprecated. Use TensorArraySplitV3

TensorArrayUnpack =>	

TensorArrayWrite =>	

TensorArrayWriteV2 =>	Deprecated. Use TensorArrayGradV3

TextLineDataset =>	Creates a dataset that emits the lines of one or more text files.

WriteFile =>	Writes contents to the file at input filename. Creates file if not existing.


