/* oo_proto.h — stub for bare-metal build
 * The actual protocol types are compiled into liboo-modules.a.
 * This stub satisfies the header-time declarations only.
 */
#pragma once
#include <stdint.h>

/* OO-NET protocol IDs */
#define OO_PROTO_MAGIC   0x4F4F4E45u  /* "OONE" */
#define OO_PROTO_VERSION 1

typedef uint32_t OoProtoMsgType;
typedef uint32_t OoNodeId;

typedef struct {
    uint32_t magic;
    uint32_t version;
    OoProtoMsgType msg_type;
    OoNodeId src;
    OoNodeId dst;
    uint32_t payload_len;
} OoProtoHeader;

#define OO_PROTO_MSG_DNA_SYNC    0x01u
#define OO_PROTO_MSG_TOKEN_BCAST 0x02u
#define OO_PROTO_MSG_DECISION    0x03u
#define OO_PROTO_MSG_HEARTBEAT   0x04u

typedef uint32_t OOEvent;
#define OO_EVENT_NONE        0x00u
#define OO_EVENT_DNA_SYNC    0x01u
#define OO_EVENT_TOKEN_BCAST 0x02u
#define OO_EVENT_DECISION    0x03u
#define OO_EVENT_HEARTBEAT   0x04u
