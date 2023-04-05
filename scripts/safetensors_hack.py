import io
import os
import mmap
import torch
import json
import hashlib
import safetensors
import safetensors.torch

from modules import sd_models

# PyTorch 1.13 and later have _UntypedStorage renamed to UntypedStorage
UntypedStorage = torch.storage.UntypedStorage if hasattr(torch.storage, 'UntypedStorage') else torch.storage._UntypedStorage
META_DATA_SUFFIX=".metadata"
INFO_DATA_SUFFIX = ".info"
INFO_DATA_SUFFIX2 = ".civitai.info"
default_meta = {
    "ssmd_cover_images": '[]',
    "ssmd_display_name": "",
    "ssmd_author": "",
    "ssmd_source": "",
    "ssmd_keywords": "",
    "ssmd_description": "",
    "ssmd_rating": "0",
    "ssmd_tags": ""
    # "sshs_model_hash": model_hash,
    # "sshs_legacy_hash": legacy_hash
  }
civital_webSite = "https://civitai.com"


def read_metadata(filename):
    """Reads the JSON metadata from a .safetensors file"""
    path_split = os.path.splitext(filename)
    file_name = path_split[0]
    meta_data_path = f'{file_name}{META_DATA_SUFFIX}'
    # info_data_path = f'{file_name}{INFO_DATA_SUFFIX}'
    info_data_path = f'{file_name}{INFO_DATA_SUFFIX2}'
    # 如果有info_data 
    if os.path.exists(info_data_path):
        metadata = None
        #如果有metadata
        if os.path.exists(meta_data_path):
            # 直接用metadata的,因为不改变info_data
            with open(meta_data_path,mode='r',encoding="utf8") as f:
                metadata = json.loads(f.read())
        if metadata:
            print("load metadata from meta_data_file")
            return metadata
        print("load metadata from info_data_file")
        with open(info_data_path,mode='r',encoding="utf8") as f:
            info_data = json.loads(f.read())
            metadata = convert_info_data_to_meta_data(info_data)
            return metadata  
    else:
        print("load metadata from model_file")
        # 如果没有 直接从模型读取
        with open(filename, mode="r", encoding="utf8") as file_obj:
            with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
                header = m.read(8)
                n = int.from_bytes(header, "little")
                metadata_bytes = m.read(n)
                metadata = json.loads(metadata_bytes)
                return metadata.get("__metadata__", {})


    
def convert_info_data_to_meta_data(info_from_helper):
    """ 转换info 到meta"""
    _meta = default_meta.copy()
    trainedWords = info_from_helper.get('trainedWords')
    if trainedWords and isinstance(trainedWords,list):
      _meta['ssmd_keywords'] = ','.join(trainedWords)
    model = info_from_helper.get('model')
    if model:
      _meta['ssmd_display_name'] = model.get('name','')
    _meta['ssmd_description'] = info_from_helper.get('description')
    _meta['ssmd_source'] = f'{civital_webSite}/models/{info_from_helper.get("id")}'
    # 如果有多余的
    _model_info = info_from_helper.get('_model_info')
    if _model_info:
      _meta['ssmd_author'] = _model_info.get('creator',{}).get('username','')
      _meta['ssmd_display_name'] = _model_info.get("name","")
      stats = _model_info.get("stats")
      if stats:
        rating = stats.get("rating")
        if rating is not None:
           _meta['ssmd_rating'] = str(rating)
      tags = _model_info.get('tags')
      if tags and isinstance(trainedWords,list):
        _meta['ssmd_tags'] = ','.join(list(map(lambda x: x['name'],tags)))
    else:
      try:
        images = info_from_helper.get('images')
        if images:
          userIds = set()
          tag_map = {}
          for img in images:
            userId = img.get('userId')
            if userId:
              userIds.add(f'userId:{userId}')
            #解析tag
            tags = img.get('tags',[])
           
            for tag in tags:
              tag_info = tag.get("tag")
              if tag_info:
                tag_info_name = tag_info.get('name')
                tag_count = tag_map.get(tag_info_name,0)
                tag_count += 1
                tag_map[tag_info_name] = tag_count
          _meta['ssmd_author'] = ','.join(list(userIds))
          #转换tag
          tags_infos = []
          for tag_info_name,cout in tag_map.items():
            tags_infos.append({
              "name":tag_info_name,
              "tag_count":cout
            })
          if tags_infos:
            # 排序tag_infos
            tags_infos_sorted = sorted(tags_infos,key=lambda x: x['tag_count'],reverse=True)
            #转换
            tags_infos_sorted_str = [f'{tag_info["name"]}({tag_info["tag_count"]})' for tag_info in tags_infos_sorted]
            if tags_infos_sorted_str:
              _meta['ssmd_tags'] = ','.join(tags_infos_sorted_str)        
      except Exception as e:
        print(e)
    return _meta

def write_metadata_to_info(filename,metadata):
    """Write the JSON metadata from model_path.metadata"""
    path_split = os.path.splitext(filename)
    file_name = path_split[0]
    meta_data_path = f'{file_name}{META_DATA_SUFFIX}'
    info_data_path = f'{file_name}{INFO_DATA_SUFFIX}'
    with open(meta_data_path,mode="w",encoding="utf8") as f:
        f.write(json.dumps(metadata))
    print(f"[MetadataEditor] meta_data saved: {meta_data_path}")
   

def load_file(filename, device):
    """"Loads a .safetensors file without memory mapping that locks the model file.
    Works around safetensors issue: https://github.com/huggingface/safetensors/issues/164"""
    metadata = None
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)
    # use read_metadata instead
    md = read_metadata(filename)
    size = os.stat(filename).st_size
    storage = UntypedStorage.from_file(filename, False, size)
    offset = n + 8
    return {name: create_tensor(storage, info, offset) for name, info in metadata.items() if name != "__metadata__"}, md


def hash_file(filename):
    """Hashes a .safetensors file using the new hashing method.
    Only hashes the weights of the model."""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")

    with open(filename, mode="rb") as file_obj:
        offset = n + 8
        file_obj.seek(offset)
        for chunk in iter(lambda: file_obj.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def legacy_hash_file(filename):
    """Hashes a model file using the legacy `sd_models.model_hash()` method."""
    hash_sha256 = hashlib.sha256()

    metadata = read_metadata(filename)

    # For compatibility with legacy models: This replicates the behavior of
    # sd_models.model_hash as if there were no user-specified metadata in the
    # .safetensors file. That leaves the training parameters, which are
    # immutable. It is important the hash does not include the embedded user
    # metadata as that would mean the hash could change every time the user
    # updates the name/description/etc. The new hashing method fixes this
    # problem by only hashing the region of the file containing the tensors.
    if any(not k.startswith("ss_") for k in metadata):
      # Strip the user metadata, re-serialize the file as if it were freshly
      # created from sd-scripts, and hash that with model_hash's behavior.
      tensors, metadata = load_file(filename, "cpu")
      metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}
      model_bytes = safetensors.torch.save(tensors, metadata)

      hash_sha256.update(model_bytes[0x100000:0x110000])
      return hash_sha256.hexdigest()[0:8]
    else:
      # This should work fine with model_hash since when the legacy hashing
      # method was being used the user metadata system hadn't been implemented
      # yet.
      return sd_models.model_hash(filename)


DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    # "U64": torch.uint64,
    "I32": torch.int32,
    # "U32": torch.uint32,
    "I16": torch.int16,
    # "U16": torch.uint16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool
}


def create_tensor(storage, info, offset):
    """Creates a tensor without holding on to an open handle to the parent model
    file."""
    dtype = DTYPES[info["dtype"]]
    shape = info["shape"]
    start, stop = info["data_offsets"]
    return torch.asarray(storage[start + offset : stop + offset], dtype=torch.uint8).view(dtype=dtype).reshape(shape).clone().detach()
