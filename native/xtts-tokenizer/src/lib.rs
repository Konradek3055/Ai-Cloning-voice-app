use jni::JNIEnv;
use jni::objects::{JByteArray, JObject, JString};
use jni::sys::{jint, jintArray, jlong};
use tokenizers::Tokenizer;

#[no_mangle]
pub extern "system" fn Java_pl_kinotts_studio_xtts_HFTokenizer_nativeCreateTokenizer(
    env: JNIEnv,
    _this: JObject,
    tokenizer_bytes: JByteArray,
) -> jlong {
    let bytes = match env.convert_byte_array(&tokenizer_bytes) {
        Ok(value) => value,
        Err(_) => return 0,
    };

    match Tokenizer::from_bytes(&bytes) {
        Ok(tokenizer) => Box::into_raw(Box::new(tokenizer)) as jlong,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "system" fn Java_pl_kinotts_studio_xtts_HFTokenizer_nativeEncode(
    env: JNIEnv,
    _this: JObject,
    tokenizer_ptr: jlong,
    text: JString,
) -> jintArray {
    if tokenizer_ptr == 0 {
        return std::ptr::null_mut();
    }

    let text: String = match env.get_string(&text) {
        Ok(value) => value.into(),
        Err(_) => return std::ptr::null_mut(),
    };

    let tokenizer = unsafe { &*(tokenizer_ptr as *const Tokenizer) };
    // This matches Tokenizer.encode(txt).ids in the model's xtts_tokenizer.py.
    let encoding = match tokenizer.encode(text, true) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };

    let ids: Vec<jint> = encoding.get_ids().iter().map(|&id| id as jint).collect();
    let output = match env.new_int_array(ids.len() as i32) {
        Ok(value) => value,
        Err(_) => return std::ptr::null_mut(),
    };

    if env.set_int_array_region(&output, 0, &ids).is_err() {
        return std::ptr::null_mut();
    }

    output.into_raw()
}

#[no_mangle]
pub extern "system" fn Java_pl_kinotts_studio_xtts_HFTokenizer_nativeTokenId(
    env: JNIEnv,
    _this: JObject,
    tokenizer_ptr: jlong,
    token: JString,
) -> jint {
    if tokenizer_ptr == 0 {
        return -1;
    }

    let token: String = match env.get_string(&token) {
        Ok(value) => value.into(),
        Err(_) => return -1,
    };

    let tokenizer = unsafe { &*(tokenizer_ptr as *const Tokenizer) };
    tokenizer.token_to_id(&token).map(|id| id as jint).unwrap_or(-1)
}

#[no_mangle]
pub extern "system" fn Java_pl_kinotts_studio_xtts_HFTokenizer_nativeDeleteTokenizer(
    _env: JNIEnv,
    _this: JObject,
    tokenizer_ptr: jlong,
) {
    if tokenizer_ptr != 0 {
        unsafe {
            drop(Box::from_raw(tokenizer_ptr as *mut Tokenizer));
        }
    }
}
