using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;

namespace ControllerSelection {
    public class laser_teleport : MonoBehaviour
    {
        public GameObject player;
        public UnityEngine.UI.Text outText;

        public void OnGroundClick(BaseEventData data) {
            //outText
            outText.text = "Hit!!!";

            // Get hit point position
            OVRRayPointerEventData pData = (OVRRayPointerEventData)data;
            Vector3 destinationPosition = Vector3.zero;
            destinationPosition = pData.pointerCurrentRaycast.worldPosition;

            // Get current position of player
            Vector3 current_player_position = player.transform.position;

            // Get updated position of player
            current_player_position.x = destinationPosition.x;
            current_player_position.z = destinationPosition.z;
            Vector3 updated_player_position = current_player_position;

            // Update position of player
            player.transform.position = updated_player_position;

        }
    }
}
