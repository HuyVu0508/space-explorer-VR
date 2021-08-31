using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class start_controller_desk : MonoBehaviour
{
    public GameObject player;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        GameObject desk_disk = transform.Find("desk_disk").gameObject;

        // If player enters disk, turn on controller desk
        if (enter_disk(player, desk_disk)){
            transform.Find("Canvas").gameObject.SetActive(true);
            transform.Find("Point Light").gameObject.SetActive(true);
        }
        else{
            transform.Find("Canvas").gameObject.SetActive(false);
            transform.Find("Point Light").gameObject.SetActive(false);            
        }

    }

    // Method for checking if player enters disk
    bool enter_disk(GameObject player, GameObject disk){
        Vector3 position1 =  player.transform.position;
        Vector3 position2 =  disk.transform.position;
        Vector3 disk_size = disk.transform.GetComponent<Renderer>().bounds.size;

        float x_distance = Mathf.Abs(position1.x - position2.x);
        float z_distance = Mathf.Abs(position1.z - position2.z);

        Debug.Log("x_distance: " + x_distance.ToString() + " / " + "z_distance: " + z_distance.ToString() + " / " + "disk_size: " + disk_size.ToString());

        if ( (x_distance < (disk_size.x/2)) && (z_distance < (disk_size.z/2)) ){
            return true;
        }
        else{
            return false;
        }
    }

}


