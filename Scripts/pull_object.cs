using UnityEngine;
using UnityEngine.AI;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;

namespace ControllerSelection {
    public class pull_object : MonoBehaviour
    {
        public float speed;
        public UnityEngine.UI.Text outText;

        // Start is called before the first frame update
        void Start()
        {
            
        }

        // Update is called once per frame
        void Update()
        {
            
        }

        // Interaction when hover
        public void OnHoverEnter(BaseEventData data){
                
            outText.text = transform.gameObject.name + " selected" ;    

        }

        // Interaction when selection
        public void OnPress(BaseEventData data){
                

            // Get OVRRayPointerEventData data
            OVRRayPointerEventData pData = (OVRRayPointerEventData)data;

            // Get ray direction
            Vector3 ray_direction = pData.worldSpaceRay.direction;

            // // Debug.Log
            // Debug.Log("Ray direction: " + ray_direction.ToString());
            // outText.text = "Ray direction: " + ray_direction.ToString();

            // Move object closer to OVRPlayerCamera
            // transform.position = transform.position + speed * (-1) * ray_direction * Time.deltaTime;
            transform.position = transform.position + speed * (-1) * ray_direction;
        }


    }
}
